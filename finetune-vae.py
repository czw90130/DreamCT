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
    vae = getLatent_model(args.pretrained_model_name_or_path)
    # Move vae to gpu
    vae.to(device, dtype=weight_dtype)  

    vae.requires_grad_(True)
#     vae_trainable_params = [] 
#     for name, param in vae.named_parameters():
#         if 'decoder' in name:
#             param.requires_grad = True
#             vae_trainable_params.append(param)
     
#     print(f"VAE total params = {len(list(vae.named_parameters()))}, trainable params = {len(vae_trainable_params)}")
    
    if args.gradient_checkpointing:
        vae.gradient_checkpointing_enable()

    optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        # itertools.chain(vae_trainable_params)
        itertools.chain(vae.parameters())
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = Sag3DDataset(root_dir=args.ct_data_dir,target_size=512, num_frames=1, num_samples_per_npz=2000, output_with_info=False) # 2D

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

    def inputs2img(input):
        target_images = (input / 2 + 0.5).clamp(0, 1)  
        target_images = target_images.detach().cpu().numpy()
        target_images = (target_images * 255).round().astype("uint8")
        return target_images
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch, args.num_train_epochs):
        vae.train()
        for batch in train_dataloader:
            slice = batch['data'][:,:,0].clamp(-1, 1).to(device, dtype=weight_dtype) # 2D VAE 深度D为1
            
            # Convert images to latent space
            latents = vae.encode(slice.to(dtype=weight_dtype)).latent_dist.sample()  
            latents = latents * 0.18215

            latents = 1 / 0.18215 * latents            
            pred_images = vae.decode(latents).sample
            pred_images = pred_images.clamp(-1, 1)

            loss = F.mse_loss(pred_images, slice, reduction="mean")
 
            loss.backward()
                        
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 50 == 0:
                with torch.no_grad():
                    pred_images = inputs2img(pred_images) 
                    target = inputs2img(slice)
                    viz = np.concatenate([pred_images[0], target[0]], axis=2)  
                    writer.add_image(f'train/pred_img', viz, global_step=global_step)

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
