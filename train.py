import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
from einops import rearrange
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__)

from utils.parse_args import parse_args
from datasets.dreamct_dataset import CTFramesDataset
from pipelines.dual_encoder_pipeline import StableDiffusionCT2CTPipeline
from models.unet_dual_encoder import *

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    writer = SummaryWriter(os.path.join(args.output_dir, args.logging_dir, args.run_name))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # # Load CLIP Image Encoder
    # clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    # clip_encoder.requires_grad_(False)
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load text encoder
    text_encoder, tokenizer = load_text_encoder(args.pretrained_model_name_or_path)
    text_encoder.requires_grad_(False)
    

    # Load models and create wrapper for stable diffusion
    vae = getLatent_model(args.pretrained_model_name_or_path)

    # Load pretrained UNet layers
    unet = get_unet(args.pretrained_model_name_or_path)

    if args.custom_chkpt is not None:
        print("Loading ", args.custom_chkpt)
        unet_state_dict = torch.load(args.custom_chkpt)
        new_state_dict = OrderedDict()
        for k, v in unet_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k 
            new_state_dict[name] = v
        unet.load_state_dict(new_state_dict)
        unet = unet.cuda()

    # Embedding adapter
    adapter = Embedding_Adapter(num_vaes=args.preframe_num)

    if args.custom_chkpt is not None:
        adapter_chkpt = args.custom_chkpt.replace('unet_epoch', 'adapter')
        print("Loading ", adapter_chkpt)
        adapter_state_dict = torch.load(adapter_chkpt)
        new_state_dict = OrderedDict()
        for k, v in adapter_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k 
            new_state_dict[name] = v
        adapter.load_state_dict(new_state_dict)
        adapter = adapter.cuda()

    #adapter.requires_grad_(True)

    vae.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        adapter.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    # if args.use_8bit_adam:
    #     try:
    #         import bitsandbytes as bnb
    #     except ImportError:
    #         raise ImportError(
    #             "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
    #         )

    #     optimizer_class = bnb.optim.AdamW8bit
    # else:
    optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), adapter.parameters(),)
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_dataset = CTFramesDataset(ct_data_root=args.ct_data_dir, frame_num=args.preframe_num)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, adapter, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the image_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    global_step = 0

    def latents2img(latents):
        latents = 1 / 0.18215 * latents
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().numpy()
        images = (images * 255).round().astype("uint8")
        return images

    def inputs2img(input):
        target_images = (input / 2 + 0.5).clamp(0, 1)
        target_images = target_images.detach().cpu().numpy()
        target_images = (target_images * 255).round().astype("uint8")
        return target_images

    # latest_chkpt_step = 0
    for epoch in range(args.epoch, args.num_train_epochs):
        unet.train()
        adapter.train()
        # first_batch = True
        for batch in train_dataloader:
            # if first_batch and latest_chkpt_step is not None:
            #     #os.system(f"python test_img2img.py --step {latest_chkpt_step} --strength 0.8")
            #     first_batch = False
            with accelerator.accumulate(unet):
                # batch[0]: frame data
                # batch[1]: properties
                # 分离前置帧和目标帧
                target_frames = batch[0][:,-1]
                target_texts = batch[1]['sentence']
                
                # Convert images to latent space
                target_latents = vae.encode(target_frames[:, :3].to(dtype=weight_dtype)).latent_dist.sample()
                target_latents = target_latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # print("target_latents shape = ", target_latents.shape)
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                # print("noisy_latents shape = ", noisy_latents.shape)

                # Concatenate spine markers with noise
                _, _, h, w = noisy_latents.shape
                noisy_latents = torch.cat((noisy_latents, F.interpolate(target_frames[:,3].unsqueeze(1), (h,w))), 1)
                # print("new noisy_latents shape = ", noisy_latents.shape)
                
                # 编码文字
                input_ids = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).input_ids
                input_ids = input_ids.to(target_latents.device)
                clip_hidden_states = text_encoder(input_ids).last_hidden_state
                #print("clip states shape = ", clip_hidden_states.shape)
                
                # Get VAE embeddings
                vae_hidden_states = []
                for i in range(args.preframe_num):
                    vae_hs = vae.encode(batch[0][:,i,:3].to(device=target_latents.device, dtype=weight_dtype)).latent_dist.sample() * 0.18215
                    # if i==0:
                    #     print("vae states shape = ", vae_hs.shape)
                    vae_hidden_states.append(vae_hs)
                
                encoder_hidden_states = adapter(clip_hidden_states, vae_hidden_states)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # write to tensorboard
            if global_step % 10 == 0:
                weights = adapter.linear1.weight.cpu().detach().numpy()
                weights = np.sum(weights, axis=0)
                weights = weights.flatten()
                plt.figure()
                plt.plot(range(len(weights)), weights)
                plt.title(f"VAE Weights = {weights[50:]}")
                #plt.hist(weights)
                writer.add_figure('embedding_weights', plt.gcf(), global_step=global_step)

            writer.add_scalar("loss/train", loss.detach().item(), global_step)
            if global_step % 50 == 0:
                with torch.no_grad():
                    pred_latents = noisy_latents[:,:4,:,:] - model_pred
                    pred_images = latents2img(pred_latents)
                    
                    noise_viz = latents2img(noisy_latents[:,:4,:,:])
                    target = inputs2img(target_frames[:,:3])
                    input_img = inputs2img(batch[0][:,-2,:3])
                    
                    writer.add_image(f'train/input_last', input_img[0], global_step=global_step)
                    writer.add_image(f'train/noise_viz', noise_viz[0], global_step=global_step)
                    writer.add_image(f'train/pred_img', pred_images[0], global_step=global_step)
                    writer.add_image(f'train/target', target[0], global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
            progress_bar.set_description(f"Epoch {epoch}/{args.num_train_epochs}|Steps {global_step}")

            if global_step >= args.max_train_steps:
                break

            # save model
            if accelerator.is_main_process and global_step % 500 == 1:
                pipeline = StableDiffusionCT2CTPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=accelerator.unwrap_model(tokenizer),
                    unet=accelerator.unwrap_model(unet),
                    # adapter=accelerator.unwrap_model(adapter),
                )
                pipeline.save_pretrained(os.path.join(args.output_dir, f'checkpoint-{epoch}'))
                model_path = os.path.join(args.output_dir, f'unet_epoch_{epoch}.pth')
                torch.save(unet.state_dict(), model_path)
                adapter_path = os.path.join(args.output_dir, f'adapter_{epoch}.pth')
                torch.save(adapter.state_dict(), adapter_path)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)