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

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='dreampose-1', help="Path to custom pretrained checkpoints folder.",)
parser.add_argument("--pose_folder", default='../UBC_Fashion_Dataset/valid/91iZ9x8NI0S.mp4', help="Path to test frames, poses, and joints.",)
parser.add_argument("--test_poses", default=None, help="Path to test frames, poses, and joints.",)
parser.add_argument("--epoch", type=int, default=44, required=True, help="Pretrained custom model checkpoint epoch number.",)
parser.add_argument("--ct_path", default='testdata/s0011.npz', help="Path to ct.",)
parser.add_argument("--pose_path", default='../UBC_Fashion_Dataset/valid/A1F1j+kNaDS.mp4/85_to_95_to_116/skeleton_i.npy', help="Pretrained model checkpoint step number.",)
parser.add_argument("--strength", type=float, default=1.0, required=False, help="How much noise to add to input image.",)
parser.add_argument("--s1", type=float, default=0.5, required=False, help="Classifier free guidance of input image.",)
parser.add_argument("--s2", type=float, default=0.5, required=False, help="Classifier free guidance of input pose.",)
parser.add_argument("--iters", default=1, type=int, help="# times to do stochastic sampling for all frames.")
parser.add_argument("--sampler", default='PNDM', help="PNDM or DDIM.")
parser.add_argument("--n_steps", default=100, type=int, help="Number of denoising steps.")
parser.add_argument("--output_dir", default=None, help="Where to save results.")
parser.add_argument("--j", type=int, default=-1, required=False, help="Specific frame number.",)
parser.add_argument("--min_j", type=int, default=0, required=False, help="Lowest predicted frame id.",)
parser.add_argument("--max_j", type=int, default=-1, required=False, help="Max predicted frame id.",)
parser.add_argument("--custom_vae", default=None, help="Path use custom VAE checkpoint.")
parser.add_argument("--batch_size", type=int, default=1, required=False, help="# frames to infer at once.",)
args = parser.parse_args()

save_folder = args.output_dir if args.output_dir is not None else args.folder #'results-fashion/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Load custom model
model_id = f"{args.folder}/checkpoint-{args.epoch}" #if args.step > 0 else "CompVis/stable-diffusion-v1-4"
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

def visualize_dp(im, dp):
        #im = im.transpose((2, 0, 1))
        print(im.shape, dp.shape)
        hsv = np.zeros(im.shape, dtype=np.uint8)
        hsv[..., 1] = 255

        dp = dp.cpu().detach().numpy()
        mag, ang = cv2.cartToPolar(dp[0], dp[1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

n_images_per_sample = 1

frame_numbers = sorted([int(path.split('frame_')[-1].replace('_densepose.npy', '')) for path in glob.glob(f'{args.pose_folder}/frame_*.npy')])
frame_numbers = list(set(frame_numbers))
pose_paths = [f'{args.pose_folder}/frame_{num}_densepose.npy' for num in frame_numbers]

if args.max_j > -1:
    pose_paths = pose_paths[args.min_j:args.max_j]
else:
    pose_paths = pose_paths[args.min_j:]

imSize = (512, 640)
image_transforms = transforms.Compose(
    [
        transforms.Resize(imSize, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
tensor_transforms = transforms.Compose(
    [
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Load ct data
npz_data = np.load(args.ct_path, allow_pickle=True)


if args.j >= 0:
    j = args.j
    pose_paths = pose_paths[j:j+1]

# Iterate samples
prev_image = input_image
for i, pose_path in enumerate(pose_paths):
    frame_number = int(frame_numbers[i])
    h, w = imSize[1], imSize[0]

    # construct 5 input poses
    poses = []
    for pose_number in range(frame_number-2, frame_number+3):
        dp_path = pose_path.replace(str(frame_number), str(pose_number))
        if not os.path.exists(dp_path):
            dp_path = pose_path
        print(dp_path)
        dp_i = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (h, w), mode='bilinear').squeeze(0)
        poses.append(tensor_transforms(dp_i))
    input_pose = torch.cat(poses, 0).unsqueeze(0)

    print(pose_path.split('_'))
    j = int(pose_path.split('_')[-2])
    print("j = ", j)

    with autocast():
        image = pipe(prompt="",
                    image=input_image,
                    pose=input_pose,
                    strength=1.0,
                    num_inference_steps=args.n_steps,
                    guidance_scale=7.5,
                    s1=args.s1,
                    s2=args.s2,
                    callback_steps=1,
                    frames=[]
                )[0][0]

        

        # Save pose and image
        save_path = f"{save_folder}/pred_#{j}.png"
        image = image.convert('RGB')
        image = np.array(image)
        image = image - np.min(image)
        image = (255*(image / np.max(image))).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    
