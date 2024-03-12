import os
import torch
import argparse
import numpy as np
import imageio
from torch.cuda.amp import autocast
import torch.nn.functional as F
from diffusers import DDIMScheduler
from einops import rearrange
from tqdm import tqdm
from models.unet_dual_encoder import *

from datasets.nifti_mat import NIfTIEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Path to custom pretrained checkpoints folder.")
parser.add_argument("--obj_dir", default='testdata/s0011/zanatomy', help="Path to obj spine marker.")  
parser.add_argument("--ct_path", default='testdata/s0011/ct.nii.gz', help="Path to ct.")
parser.add_argument("--n_steps", default=100, type=int, help="Number of denoising steps.")
parser.add_argument("--output_dir", default='testdata/result', help="Where to save results.")
args = parser.parse_args()

device = "cuda"

# Load models
unet = get_unet3d(args.folder).to(device)
vae = getLatent_model(args.folder).to(device)
text_encoder, tokenizer = load_text_encoder(args.folder)
scheduler = DDIMScheduler.from_pretrained(args.folder, subfolder="scheduler")

# Prepare pipeline 
class StableDiffusionCT2CTPipeline:
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def __call__(self, prompt, masked_frames, mask_frames, num_inference_steps):
        device = self.unet.device
        
        # Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
            masked_frames = masked_frames.unsqueeze(0)
            mask_frames = mask_frames.unsqueeze(0)
        else:
            batch_size = len(prompt)
        
        frames_num = masked_frames.shape[1]
        
        # Encode prompt
        with autocast():
            # 编码文字
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
            input_ids = input_ids.to(device)
            embeddings = self.text_encoder(input_ids).last_hidden_state
            
        global_dtype = embeddings.dtype
        
        masked_frames = masked_frames.to(device, dtype=global_dtype)
        mask_frames = mask_frames.to(device, dtype=global_dtype)
        
        # Preprocess image
        init_frames = masked_frames.clone()

        # Set timesteps    
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        with autocast():
            target_latents = []
            masked_latents = []
            for i in range(frames_num):
                target_latent = self.vae.encode(init_frames[:, i].to(dtype=global_dtype)).latent_dist.sample()
                target_latent = target_latent * 0.18215
                target_latents.append(target_latent)

                masked_latent = self.vae.encode(masked_frames[:, i].to(dtype=global_dtype)).latent_dist.sample()
                masked_latent = masked_latent * 0.18215
                masked_latents.append(masked_latent)
            
            target_latents = torch.stack(target_latents, dim=1).permute(0,2,1,3,4)
            masked_latents = torch.stack(masked_latents, dim=1).permute(0,2,1,3,4)
            
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(target_latents)
            bsz = target_latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device, dtype=global_dtype)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(target_latents, noise, timesteps[0].repeat(batch_size))

            _, _, f, h, w = noisy_latents.shape
            mask_frames = F.interpolate(mask_frames, size=(f,h,w)).to(dtype=global_dtype)

            # Prepare input latents and additional info
            input_latents = torch.cat([noisy_latents, mask_frames, masked_latents], dim=1).to(dtype=global_dtype)
        
        # Denoising loop
        for t in tqdm(timesteps):
            latent_model_input = self.scheduler.scale_model_input(input_latents, t)

            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embeddings)["sample"]

            # Get the target for loss depending on the prediction type
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise 
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(input_latents, noise, t)
            else:
                raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
            
            # Compute the previous noisy sample x_t -> x_t-1
            input_latents = self.scheduler.step(noise_pred, t, input_latents)["prev_sample"]
        
        # Decode latents
        latents = input_latents[:,:4,:,:,:]
        latents = 1 / 0.18215 * latents
        frames = self.vae.decode(latents).sample.clamp(-1, 1)

        return frames
        
pipe = StableDiffusionCT2CTPipeline(vae, text_encoder, tokenizer, unet, scheduler)

if __name__ == "__main__":
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    
    # Load data
    encoder = NIfTIEncoder()
    ct = encoder(args.ct_path, args.obj_dir, needs_extension=True)
    
    image_buff = []
    for i in tqdm(range(0, 7)):
        masked_frames, mask_frames, properties = encoder.to_frames('sag_slices', i*32, 32, slice_size=512, crop=0, mask_edge=20, randomize_sentence=False, random_cat=False)
        
        # Create masked image
        before_img = masked_frames[:, :, :3].permute(0, 2, 3, 1).cpu().numpy()  
        masked_frames[:, :, :3] = masked_frames[:, :, :3] * (mask_frames.unsqueeze(2) < 0.5)  # 反向掩码
        masked_frames[:, :, 1] = masked_frames[:, :, 3]  # 添加标记
        masked_frames[:, :, 2] = masked_frames[:, :, 3]  # 添加标记

        # Generate image  
        image = pipe(prompt=properties['sentence'], masked_frames=masked_frames, mask_frames=mask_frames, num_inference_steps=args.n_steps)
        
        middle_img = masked_frames[:, :, :3].permute(0, 2, 3, 1).cpu().numpy()
        after_img = image[:, :, :3].permute(0, 2, 3, 1).cpu().numpy()
        
        for j in range(32):
            combine_image = np.concatenate((before_img[j], middle_img[j], after_img[j]), axis=1)
            combine_image = ((combine_image + 1) / 2 * 255).astype(np.uint8)
            image_buff.append(combine_image)
        
    imageio.mimsave(os.path.join(save_folder, 'concatenated.gif'), image_buff, fps=5)