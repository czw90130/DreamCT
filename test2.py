import os
import torch
import argparse
import numpy as np
import imageio
from  torch.cuda.amp import autocast
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
# Load UNet
unet = get_unet(args.folder, "unet").to(device)
# Load VAE
vae = getLatent_model(args.folder, "vae").to(device)
# Load text encoder and tokenizer
text_encoder, tokenizer = load_text_encoder(args.folder)
# Load scheduler
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
    def __call__(self, prompt, masked_img, mask, num_inference_steps):
        device = self.unet.device
        
        # Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
            masked_img = masked_img.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            batch_size = len(prompt)
        
        # Encode prompt
        with autocast():
            # 编码文字
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
            input_ids = input_ids.cuda()
            embeddings = self.text_encoder(input_ids).last_hidden_state
            
        global_dtype = embeddings.dtype
        
        masked_img = masked_img.to(device, dtype=global_dtype)
        
        # Preprocess image
        init_image = masked_img.clone()

        # Set timesteps    
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        with autocast():
            init_latents = self.vae.encode(init_image.to(device)).latent_dist.sample()
            init_latents = 0.18215 * init_latents
            
            # add noise to latents using the timesteps
            #print("Latents Shape = ", init_latents.shape, init_latents[0].shape, image.shape[0])
            single_fixed_noise = torch.randn(init_latents[0].shape, device=device, dtype=global_dtype)
            noise = single_fixed_noise.repeat(init_image.shape[0], 1, 1, 1)#torch.tensor([single_fixed_noise for _ in range(image.shape[0])])

            init_latents = self.scheduler.add_noise(init_latents.cuda(), noise.cuda(), timesteps[0].repeat(batch_size))
            latents = init_latents
            
            masked_latent_disk = self.vae.encode(masked_img).latent_dist
            masked_latents = masked_latent_disk.sample()
            masked_latents = 0.18215 * masked_latents
            
            
            
        _, _, h, w = latents.shape
        
        # Denoising loop
        for t in tqdm(timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            # Add spine_marker to noisy latents
            latent_model_input = torch.cat((latent_model_input.cuda(), F.interpolate(mask, (h,w)).cuda(), masked_latents.cuda()), 1)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embeddings.cuda())["sample"]

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        # Decode latents
        latents = latents[:,:4, :, :]
        latents = 1 / 0.18215 * latents        
        image = self.vae.decode(latents).sample.clamp(-1, 1)

        return image
        
pipe = StableDiffusionCT2CTPipeline(vae, text_encoder, tokenizer, unet, scheduler)

if __name__ == "__main__":
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    
    # Load data
    encoder = NIfTIEncoder()
    ct = encoder(args.ct_path, args.obj_dir, needs_extension=True)
    
    image_buff = []
    for i in tqdm(range(0, 7)):
        slice, mask, properties = encoder.to_slice('sag_slices', i, slice_size=512, crop=0, mask_edge=20, randomize_sentence=False, random_cat=False)
        mask = mask.unsqueeze(0)
        
        # Create masked image
        before_img = slice[:3].permute(1,2,0).cpu().numpy()  
        masked_img = slice[:3] * (mask < 0.5)
        masked_img[1] = slice[3]
        masked_img[2] = slice[3]
        
        # Generate image
        image = pipe(prompt=properties['sentence'], masked_img=masked_img, mask=mask, num_inference_steps=args.n_steps)
        
        middle_img = masked_img.permute(1,2,0).cpu().numpy()
        after_img = image[0].permute(1,2,0).cpu().numpy()
        combine_image = np.concatenate((before_img, middle_img, after_img), axis=1)
        combine_image = ((combine_image + 1) / 2 * 255).astype(np.uint8)
        image_buff.append(combine_image)
        
    imageio.mimsave(os.path.join(save_folder, 'concatenated.gif'), image_buff, fps=2)