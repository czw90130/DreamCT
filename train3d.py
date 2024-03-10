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
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__)

from utils.parse_args import parse_args
from datasets.dreamct_dataset import CTFramesDataset
from pipelines.dual_encoder_pipeline import StableDiffusionCT2CTPipeline
from models.unet_dual_encoder import *

def main(args):
    logging_dir = Path(args.logging_dir)
    writer = SummaryWriter(os.path.join(args.logging_dir, args.run_name))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    text_encoder, tokenizer = load_text_encoder(args.pretrained_model_name_or_path)
    text_encoder.requires_grad_(False)

    vae = getLatent_model(args.pretrained_model_name_or_path)

    unet = get_unet3d(args.pretrained_model_name_or_path)
    if args.load_2d_weights:
        unet2d = get_unet(args.pretrained_model_name_or_path)
        convert_2d_to_3d_unet(unet2d, unet)

    vae.requires_grad_(False)
    vae_trainable_params = []
    for name, param in vae.named_parameters():
        if 'decoder' in name:
            param.requires_grad = True
            vae_trainable_params.append(param)

    print(f"VAE total params = {len(list(vae.named_parameters()))}, trainable params = {len(vae_trainable_params)}")
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), vae_trainable_params)
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_dataset = CTFramesDataset(ct_data_root=args.ct_data_dir, frame_num=args.frame_num)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=np.min([args.train_batch_size, 8])
    )

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

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
        
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Num accelerator processes = {accelerator.num_processes}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

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

    for epoch in range(args.epoch, args.num_train_epochs):
        unet.train()
        vae.train()
        first_batch = True
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # 分离输入帧和目标帧
                frames = batch[0]  # (batch_size, channel, frame_num, height, width)
                masks = batch[1]   # (batch_size, frame_num, height, width)
                properties = batch[2]

                target_frames = frames[:, :3]
                masked_frame = target_frames * (masks < 0.5)  # 反向掩码
                masked_frame[:, :, 1] = frames[:, 3]  # 添加标记
                masked_frame[:, :, 2] = frames[:, 3]  # 添加标记

                target_texts = properties['sentence']
                
                # Convert images to latent space
                target_latents = []
                masked_latents = []
    
                for i in range(frames.shape[1]):
                    target_latent = vae.encode(target_frames[:, i].to(dtype=weight_dtype)).latent_dist.sample()
                    target_latent = target_latent * 0.18215
                    target_latents.append(target_latent)

                    masked_latent = vae.encode(masked_frame[:, i].to(dtype=weight_dtype)).latent_dist.sample()
                    masked_latent = masked_latent * 0.18215
                    masked_latents.append(masked_latent)
                
                target_latents = torch.stack(target_latents, dim=1)
                masked_latents = torch.stack(masked_latents, dim=1)
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # Prepare input latents and additional info
                input_latents = torch.cat([noisy_latents, masks.unsqueeze(1), masked_latents], dim=2)
                
                # 编码文字
                input_ids = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).input_ids
                input_ids = input_ids.to(target_latents.device)
                clip_hidden_states = text_encoder(input_ids).last_hidden_state

                # Predict the noise residual
                model_pred = unet(input_latents, timesteps, clip_hidden_states).sample

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
                        itertools.chain(unet.parameters(), vae_trainable_params)
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if global_step % 50 == 0:
                with torch.no_grad():
                    pred_latents = noisy_latents - model_pred
                    pred_images = latents2img(pred_latents[:, -1])
                    tgt_latents = noisy_latents - target
                    tgt_images = latents2img(tgt_latents[:, -1])
                    
                    noise_viz = latents2img(noisy_latents[:, -1])
                    
                    target = inputs2img(target_frames[:, -1])
                    
                    writer.add_image(f'train/0noise_viz', noise_viz[0], global_step=global_step)
                    writer.add_image(f'train/1pred_img', pred_images[0], global_step=global_step)
                    writer.add_image(f'train/2tgt_img', tgt_images[0], global_step=global_step)
                    writer.add_image(f'train/3target', target[0], global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process and global_step % 500 == 0 and not first_batch:
                progress_bar.set_description(f"saveing models: ")
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch}')
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                progress_bar.set_description(f"saveing models: unet.pth")
                model_path = os.path.join(checkpoint_path, 'unet.pth')
                torch.save(unet.state_dict(), model_path)
                progress_bar.set_description("")
            first_batch = False
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)