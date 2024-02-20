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
from datasets.dreamct_dataset import CTdataProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Path to custom pretrained checkpoints folder.",)
parser.add_argument("--test_poses", default=None, help="Path to test frames, poses, and joints.",)
parser.add_argument("--ct_path", default='testdata/s0011.npz', help="Path to ct.",)
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

# Load adapter
adapter = Embedding_Adapter(chkpt=os.path.join(args.folder, 'adapter.pth'))

# Load scheduler
scheduler = DDIMScheduler.from_pretrained(args.folder, subfolder="scheduler")

# Load pipeline
pipe = StableDiffusionCT2CTPipeline(vae, text_encoder, tokenizer, unet, adapter, scheduler)

#pipe.unet.load_state_dict(torch.load(f'{save_folder}/unet_epoch_{args.epoch}.pth'))  #'results/epoch_1/unet.pth'))
#pipe.unet = pipe.unet.cuda()

# Change scheduler
if args.sampler == 'DDIM':
    print("Default scheduler = ", pipe.scheduler)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print("New scheduler = ", pipe.scheduler)

def visualize_im(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))

    image = np.array(image)
    image = (image + 1) / 2 * 255
    image = image.astype(np.uint8)
    return image

# Load ct data
ct_data_processor = CTdataProcessor()
frames, properties = ct_data_processor(args.ct_path, 'sag_slices', 4, slice_size=512, sample_num=4, randomize_sentence=False)

# Iterate samples
print(frames.shape)
prev_frames = frames[:-1]
target_frame = frames[-1]
spine_marker = target_frame[-1]

with autocast():
    image = pipe(prompt=properties['sentence'],
                prev_frames=prev_frames,
                spine_marker=spine_marker,
                strength=1.0,
                num_inference_steps=args.n_steps,
                guidance_scale=7.5,
                s1=args.s1,
                s2=args.s2,
                callback_steps=1,
            )[0]

    # Save pose and image
    pred_image = visualize_im(image)
    tgt_image = visualize_im(target_frame[:3])
    spine_image = visualize_im(spine_marker.unsqueeze(0))
    cv2.imwrite(os.path.join(save_folder, 'pred.png'), cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_folder, 'tgt.png'), cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_folder, 'spine.png'), cv2.cvtColor(spine_image, cv2.COLOR_BGR2RGB))


    
