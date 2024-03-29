import os
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from pipelines.dual_encoder_pipeline import StableDiffusionCT2CTPipeline
import argparse
from torchvision import transforms
import torch
import cv2, PIL, glob, random
import numpy as np
from  torch.cuda.amp import autocast
from torchvision import transforms
from collections import OrderedDict
from torch import nn
import torch, cv2
import torch.nn.functional as F
from models.unet_dual_encoder import *
from datasets.nifti_mat import NIfTIEncoder, combine_image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Path to custom pretrained checkpoints folder.",)
parser.add_argument("--obj_dir", default='testdata/s0011/zanatomy', help="Path to obj spine marker.",)
parser.add_argument("--ct_path", default='testdata/s0011/ct.nii.gz', help="Path to ct.",)
parser.add_argument("--strength", type=float, default=1.0, required=False, help="How much noise to add to input image.",)
parser.add_argument("--s1", type=float, default=0.5, required=False, help="Classifier free guidance of input image.",)
parser.add_argument("--s2", type=float, default=0.5, required=False, help="Classifier free guidance of input pose.",)
parser.add_argument("--iters", default=1, type=int, help="# times to do stochastic sampling for all frames.")
parser.add_argument("--sampler", default='PNDM', help="PNDM or DDIM.")
parser.add_argument("--n_steps", default=100, type=int, help="Number of denoising steps.")
parser.add_argument("--output_dir", default='testdata/result', help="Where to save results.")
parser.add_argument("--custom_vae", default=None, help="Path use custom VAE checkpoint.")
parser.add_argument("--batch_size", type=int, default=1, required=False, help="# frames to infer at once.",)
args = parser.parse_args()

save_folder = args.output_dir if args.output_dir is not None else args.folder #'results-fashion/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

device = "cuda"

# Load UNet
unet = get_unet(args.folder, "unet")
unet = unet.cuda()

# Load VAE
vae = getLatent_model(args.folder, "vae")

# Load text encoder and tokenizer
text_encoder, tokenizer = load_text_encoder(args.folder)

# Load scheduler
scheduler = DDIMScheduler.from_pretrained(args.folder, subfolder="scheduler")

# Load pipeline
pipe = StableDiffusionCT2CTPipeline(vae, text_encoder, tokenizer, unet, scheduler)

# Change scheduler
if args.sampler == 'DDIM':
    print("Default scheduler = ", pipe.scheduler)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print("New scheduler = ", pipe.scheduler)

if __name__ == "__main__":
    from tqdm import tqdm
    import imageio
    from copy import deepcopy
    # Load ct data
    nifti_path = args.ct_path
    obj_input_dir = args.obj_dir

    encoder = NIfTIEncoder()
    ct = encoder(nifti_path, obj_input_dir, needs_extension=True)
    # encoder.load_npz(nifti_path)
    # ct_origin = ct['meta']['origin_in_ext']
    # ct_nifti_shape = ct['meta']['nifti_shape']
    # num = 10
    
    image_buff = []
    for i in tqdm(range(0,20)):
        slice, mask, properties = encoder.to_slice('sag_slices', i, slice_size=512, crop=0, mask_edge=20, randomize_sentence=False, random_cat=False)
        mask = mask.unsqueeze(0)
        # 应用遮罩生成被遮挡的图像版本
        before_img = slice[:3].permute(1,2,0).cpu().numpy()
        masked_img = slice[:3] * (mask < 0.5)  # 反向掩码
        # 添加标记
        masked_img[1]= slice[3]
        masked_img[2] = slice[3]


        
        with autocast():
            image = pipe(prompt=properties['sentence'],
                        masked_img=masked_img,
                        mask=mask,
                        strength=1.0,
                        num_inference_steps=args.n_steps,
                        guidance_scale=7.5,
                        s1=args.s1,
                        s2=args.s2,
                        callback_steps=1,
                    )

        middle_img = masked_img.permute(1,2,0).cpu().numpy()
        # imageio.imsave(os.path.join(save_folder, 'before_image.png'), before_img)
        after_img = image[0].permute(1,2,0).cpu().numpy()
        combine_image = np.concatenate((before_img, middle_img, after_img), axis=1)
        combine_image = ((combine_image + 1) / 2 * 255).astype(np.uint8)
        image_buff.append(combine_image)
    # save image
    imageio.mimsave(os.path.join(save_folder, 'concatenated.gif'), image_buff, fps=2)
    
    
    
    
    
    # before_frame_imgs = []
    # after_frame_imgs = []
    # # Iterate samples
    
    # pbar = tqdm(range(num))
    # for i in pbar:
    #     frames, properties = encoder.to_frame('hor_slices', i,sample_num=2, slice_size=512, randomize_sentence=True, cat_prev_frame=False)
    #     prev_frames = frames
    #     target_frame = frames[-1]
    #     spine_marker = target_frame[-1]
    #     before_frame_imgs.append(combine_image(target_frame, ct['meta']))

    #     with autocast():
    #         image = pipe(prompt=properties['sentence'],
    #                     prev_frames=prev_frames,
    #                     spine_marker=spine_marker,
    #                     strength=1.0,
    #                     num_inference_steps=args.n_steps,
    #                     guidance_scale=7.5,
    #                     s1=args.s1,
    #                     s2=args.s2,
    #                     callback_steps=1,
    #                 )[0]

    #         # Save pose and image
    #         target_frame[:3] = image
    #     after_frame_imgs.append(combine_image(target_frame, ct['meta']))
    #     if i % 10 == 0 and i > 0:

    #         # Concatenate the images
    #         concatenated_images = [np.concatenate((before, after), axis=1) for before, after in zip(before_frame_imgs, after_frame_imgs)][:i]
    #         imageio.mimsave(os.path.join(save_folder, 'concatenated.gif'), concatenated_images, fps=2)
    
    # # Concatenate the images
    # concatenated_images = [np.concatenate((before, after), axis=1) for before, after in zip(before_frame_imgs, after_frame_imgs)]
    # imageio.mimsave(os.path.join(save_folder, 'concatenated.gif'), concatenated_images, fps=2)