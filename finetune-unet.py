import itertools
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np

from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils.parse_args import parse_args
from datasets.sag3d import Sag3DDataset
from models.unet_dual_encoder import *

def main(args):
    logging_dir = Path(args.logging_dir)  
    writer = SummaryWriter(os.path.join(args.logging_dir, args.run_name))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    weight_dtype = torch.float32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load models and create wrapper for stable diffusion
    # Load text encoder
    text_encoder, tokenizer = load_text_encoder(args.pretrained_model_name_or_path)
    text_encoder.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Load VAE
    vae = getLatent_model(args.pretrained_model_name_or_path).to(device, dtype=weight_dtype)
    # Load unet
    unet = get_unet(args.pretrained_model_name_or_path).to(device, dtype=weight_dtype)

    unet.requires_grad_(True)
    
    if args.gradient_checkpointing:
        unet.gradient_checkpointing_enable()

    optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters())
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = Sag3DDataset(root_dir=args.ct_data_dir, target_size=512, num_frames=1, output_mode='random', num_samples_per_npz=2000, output_with_info=True) # 2D

    train_dataloader = torch.utils.data.DataLoader(  
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=np.min([args.train_batch_size, 8])
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)) 
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size

    print("***** Running training *****")  
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")  
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Batch size per device = {args.train_batch_size}")    
    print(f"  Total train batch size = {total_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps))

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
        for batch in train_dataloader:
            slice = batch['data'][:,:,0].clamp(-1, 1).to(device, dtype=weight_dtype) # 2D VAE 深度D为1
            properties = batch['info']

            target_texts = properties['sentence']

            
            # Convert images to latent space
            latents = vae.encode(slice.to(dtype=weight_dtype)).latent_dist.sample()  
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            # print("latents shape = ", latents.shape)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # print("noisy_latents shape = ", noisy_latents.shape)

            # 编码文字
            input_ids = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).input_ids
            input_ids = input_ids.to(latents.device)
            clip_hidden_states = text_encoder(input_ids).last_hidden_state
            #print("clip states shape = ", clip_hidden_states.shape)

            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, clip_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
 
            loss.backward()
                        
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 50 == 0:
                with torch.no_grad():
                    origin_images = inputs2img(slice)
                    noise_viz = latents2img(noisy_latents)
                    # compute the previous noisy sample x_t -> x_t-1
                    target_latents = noise_scheduler.step(target, timesteps, noisy_latents)["prev_sample"]
                    target_images = latents2img(target_latents)
                    pred_latents = noise_scheduler.step(noise_pred, timesteps, noisy_latents)["prev_sample"]
                    pred_images = latents2img(pred_latents) 
                    writer.add_image(f'train/origin_image', origin_images[0], global_step=global_step)
                    writer.add_image(f'train/noise_image', noise_viz[0], global_step=global_step)
                    writer.add_image(f'train/target_image', target_images[0], global_step=global_step)
                    writer.add_image(f'train/pred_image', pred_images[0], global_step=global_step)
                    
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            progress_bar.update(1)  
            global_step += 1

            if global_step >= args.max_train_steps:
                break
           
            # save model            
            if global_step % 500 == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch}') 
                os.makedirs(checkpoint_path, exist_ok=True)

                vae_path = os.path.join(checkpoint_path,'vae.pth')  
                torch.save(vae.state_dict(), vae_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
