import itertools
import math
import os
import random
from pathlib import Path

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
from datasets.sag3d import Sag3DDataset
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)  

    # Load models and create wrapper for stable diffusion
    vae = getLatent_model(args.pretrained_model_name_or_path)

    vae.requires_grad_(False)
    vae_trainable_params = []
    for name, param in vae.named_parameters():
        if 'decoder' in name:
            param.requires_grad = True
            vae_trainable_params.append(param)

    print(f"VAE total params = {len(list(vae.named_parameters()))}, trainable params = {len(vae_trainable_params)}")
    
    if args.gradient_checkpointing:
        vae.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(vae_trainable_params)
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = Sag3DDataset(root_dir=args.ct_data_dir, num_frames=1, num_samples_per_npz=1000, output_with_info=False) # 2D

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=np.min([args.train_batch_size, 8])
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

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
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
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Num accelerator processes = {accelerator.num_processes}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    global_step = 0

    def inputs2img(input):
        target_images = (input / 2 + 0.5).clamp(0, 1)
        target_images = target_images.detach().cpu().numpy()
        target_images = (target_images * 255).round().astype("uint8")
        return target_images

    # latest_chkpt_step = 0
    for epoch in range(args.epoch, args.num_train_epochs):
        vae.train()
        first_batch = True
        for batch in train_dataloader:
            with accelerator.accumulate(vae):
                slice = batch['data'][:,:,0].clamp(-1, 1).to(accelerator.device, dtype=weight_dtype) # 2D VAE 深度D为1
                
                # Convert images to latent space
                latents = vae.encode(slice.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                latents = 1 / 0.18215 * latents
                pred_images = vae.decode(latents).sample
                pred_images = pred_images.clamp(-1, 1)

                loss = F.mse_loss(pred_images, slice, reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(vae.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if global_step % 50 == 0:
                with torch.no_grad():
                    pred_images = inputs2img(pred_images)
                    target = inputs2img(slice)
                    viz = np.concatenate([pred_images[0], target[0]], axis=2)
                    writer.add_image(f'train/pred_img', viz, global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients: 
                progress_bar.update(1)
                global_step += 1

            if global_step >= args.max_train_steps:
                break

            # save model
            if accelerator.is_main_process and global_step % 500 == 0 and not first_batch:
                progress_bar.set_description(f"saveing models: ")
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch}')
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

                progress_bar.set_description("saveing models: vae.pth")
                vae_path = os.path.join(checkpoint_path,'vae.pth')
                torch.save(vae.state_dict(), vae_path)
                progress_bar.set_description("")
            first_batch = False
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)