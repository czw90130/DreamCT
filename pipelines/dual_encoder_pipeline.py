import inspect
from typing import Callable, List, Optional, Union
from einops import rearrange

import numpy as np
import torch, torchvision
import torch.nn.functional as F
from  torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.utils import make_grid

import PIL
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor

from diffusers.configuration_utils import FrozenDict
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from models.unet_dual_encoder import get_unet, Embedding_Adapter

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusionCT2CTPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-guided image to image generation using Stable Diffusion.
    使用稳定扩散进行文本引导的图像到图像生成的管道。
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以了解库为所有管道实现的通用方法（例如下载或保存，运行在特定设备上等）。
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
            用于将图像编码和解码为潜在表示的变分自动编码器(VAE)模型。
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
            冻结的文本编码器。稳定扩散使用 CLIP 的文本部分，
            具体来说是 clip-vit-large-patch14 变体。
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            CLIPTokenizer 类的分词器。
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the encoded image latents.
            用于去噪编码的图像潜变量的条件 U-Net 架构。
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
            与 `unet` 结合使用的调度器，用于去噪编码的图像潜变量。可以是 [`DDIMScheduler`]、[`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
    """
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.__init__
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        # adapter: Embedding_Adapter,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        preframe_num: int = 3,
        stochastic_sampling: bool = False
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)
            
        self.preframe_num = preframe_num

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            # adapter=adapter
        )

        self.vae = self.vae.cuda()
        self.unet = self.unet.cuda()
        self.text_encoder = self.text_encoder.cuda()
        
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.fixed_noise = None
        self.stochastic_sampling = stochastic_sampling

        print("Stochastic Sampling: ", self.stochastic_sampling)
        
    def init_adapter(self, adapter: Embedding_Adapter):
        self.adapter = adapter.cuda()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.image_encoder, self.clip_processor, self.vae, self.adapter]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_frames(self, prompt, frames, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.
        将提示编码为文本编码器的隐藏状态。
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
                需要被编码的提示
            frames: (`torch.FloatTensor`):
                The frames to be encoded
                需要被编码的帧
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
                每个提示应生成的图像数量
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
                是否使用无分类器指导
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
                不用于指导图像生成的提示或提示。当不使用指导时忽略
                （即，如果 `guidance_scale` 小于 `1`，则忽略）。
        """
        

        with autocast():
            # 编码文字
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
            input_ids = input_ids.cuda()
            clip_hidden_states = self.text_encoder(input_ids).last_hidden_state
            #print("clip states shape = ", clip_hidden_states.shape)
            
            uncond_input_ids = self.tokenizer(negative_prompt, return_tensors="pt", padding=True, truncation=True).input_ids
            uncond_input_ids = uncond_input_ids.cuda()
            clip_uncond_hidden_states = self.text_encoder(uncond_input_ids).last_hidden_state
            
    
            uncond_frames = torch.zeros_like(frames)
            # Get VAE embeddings
            vae_hidden_states = []
            vae_uncond_hidden_states = []
            for i in range(self.preframe_num):
                vae_hs = self.vae.encode(frames[:,i,:3].cuda().float()).latent_dist.sample() * 0.18215
                # if i==0:
                #     print("vae states shape = ", vae_hs.shape)
                vae_hidden_states.append(vae_hs)
                vae_uncond_hs = self.vae.encode(uncond_frames[:,i,:3].cuda().float()).latent_dist.sample() * 0.18215
                vae_uncond_hidden_states.append(vae_uncond_hs)

            # adapt embeddings
            ct_embeddings = self.adapter(clip_hidden_states, vae_hidden_states)
            uncond_ct_embeddings = self.adapter(clip_uncond_hidden_states, vae_uncond_hidden_states)

        #print(ct_embeddings.shape)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = ct_embeddings.shape
        ct_embeddings  = ct_embeddings.repeat(1, num_images_per_prompt, 1)
        ct_embeddings = ct_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        bs_embed, seq_len, _ = uncond_ct_embeddings .shape
        uncond_ct_embeddings  = uncond_ct_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_ct_embeddings = uncond_ct_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            ct_embeddings = torch.cat([uncond_ct_embeddings, ct_embeddings, ct_embeddings])

        return ct_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        with autocast():
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, strength, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [1.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        """
        Prepare latents for the diffusion process.
        """
        with autocast():
            image = image.to(device=device, dtype=dtype).cuda()
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timesteps
        if self.fixed_noise is None:
            #print("Latents Shape = ", init_latents.shape, init_latents[0].shape, image.shape[0])
            single_fixed_noise = torch.randn(init_latents[0].shape, generator=generator, device=device, dtype=dtype)
            self.fixed_noise = single_fixed_noise.repeat(image.shape[0], 1, 1, 1)#torch.tensor([single_fixed_noise for _ in range(image.shape[0])])
        noise = self.fixed_noise

        # get latents
        init_latents = self.scheduler.add_noise(init_latents.cuda(), noise.cuda(), timestep)
        latents = init_latents

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prev_frames: torch.FloatTensor = None,
        spine_marker: torch.FloatTensor = None,
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 100,
        guidance_scale: Optional[float] = 7.5,
        s1: float = 1.0, # strength of input spine_marker
        s2: float = 1.0, # strength of input prev_frames
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        sweep = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        调用管道进行生成时会调用的函数。
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
                用于指导图像生成的提示或提示。
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
                `Image`，或表示图像批次的张量，将被用作过程的起点。
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
                从概念上讲，表示要转换参考 `image` 的程度。必须在 0 和 1 之间。`image`
                将被用作起点，添加的噪声越大，`strength` 越大。去噪步骤的数量取决于最初添加的噪声量。当 `strength` 为 1 时，添加的噪声将
                是最大的，去噪过程将运行完在 `num_inference_steps` 中指定的全部迭代次数。因此，值为 1 实质上忽略了 `image`。
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
                去噪步骤的数量。更多的去噪步骤通常会以推断速度较慢为代价，导致图像质量更高。此参数将由 `strength` 调节。
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
                如 Classifier-Free Diffusion Guidance 中定义的指导比例。
                `guidance_scale` 定义为 Imagen Paper 的方程 2 的 `w`。通过设置 `guidance_scale > 1` 启用指导比例。
                更高的指导比例通常会以图像质量较低为代价，鼓励生成与文本 `prompt` 密切相关的图像。
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
                不用于指导图像生成的提示或提示。当不使用指导时忽略（即，如果 `guidance_scale` 小于 `1`，则忽略）。
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
                每个提示要生成的图像数量。
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
                对应于 DDIM 论文中的参数 eta (η)。只适用于
                [`schedulers.DDIMScheduler`]，其他的将被忽略。
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
                一个 torch generator 使生成确定性。
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
                生成图像的输出格式。在 `PIL.Image.Image` 或 `np.array` 之间选择。
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
                是否返回 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] 而不是一个普通的元组。
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
                在推断过程中每 `callback_steps` 步将调用的函数。该函数将被以下参数调用：`callback(step: int, timestep: int, latents: torch.FloatTensor)`。
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
                `callback` 函数将被调用的频率。如果未指定，回调将在每一步被调用。
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `image_list`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `image_list`.
            When returning a list with the generated images.
            如果 `return_dict` 为 True，则返回 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]，否则返回一个包含生成的图像的列。
        """

        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Set adapter
        if adapter is not None:
            print("Setting adapter")
            self.adapter = adapter

        # 3. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 or s1 > 0.0 or s2 > 0.0

        # 4. Encode input image: [unconditional, condional, conditional]
        
        embeddings = self._encode_frames(
            prompt, prev_frames, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 5. Preprocess image
        last_frame = prev_frames[:,-1]


        # 6. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        latents = self.prepare_latents(
            last_frame, latent_timestep, batch_size, num_images_per_prompt, embeddings.dtype, device, generator
        )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. If sweeping (s1, s2) values, prepare variables
        if sweep:
            s1_vals = [0, 3, 5, 7, 9] 
            s2_vals = [0, 3, 5, 7, 9]
            images = [] # store frames
        else:
            s1_vals, s2_vals = [s1], [s2]

        # 10. Denoising loop
        copy_latents = latents.clone()
        for s1 in s1_vals:
            for s2 in s2_vals:
                latents = copy_latents.clone()
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        t = t.cuda()

                        # expand the latents if we are doing classifier free guidance
                        # 如果我们正在进行无分类器指导，则扩展潜变量
                        latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # Add spine_marker to noisy latents
                        _, _, h, w = latent_model_input.shape
                        if do_classifier_free_guidance:
                            spine_marker_input = torch.cat([torch.zeros(spine_marker.shape), spine_marker, torch.zeros(spine_marker.shape)]) 
                        else:
                            spine_marker_input = torch.cat([spine_marker, spine_marker, spine_marker]) 
                        latent_model_input = torch.cat((latent_model_input.cuda(), F.interpolate(spine_marker_input, (h,w)).cuda()), 1)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embeddings.cuda()).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            #print(f"s1={s1}, s2={s2}")
                            noise_pred_uncond, noise_pred_, noise_pred_img_only = noise_pred.chunk(3)
                            noise_pred = noise_pred_uncond + \
                                         s1 * (noise_pred_img_only - noise_pred_uncond) + \
                                         s2 * (noise_pred_ - noise_pred_img_only) 

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred.cuda(), t, latents.cuda(), **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

                # 11. Post-processing
                latents = latents[:,:4, :, :].cuda() #.float()
                image = self.decode_latents(latents)

                #print(len(image)) # 1
                #print(image[0].shape) # 640, 512, 3

                # 13. Convert to PIL
                if output_type == "pil":
                    image = self.numpy_to_pil(image)

                if sweep:
                    images.append(torchvision.transforms.ToTensor()(image[0]).clone())

        # 13. If sweeping, convert images to grid
        if sweep:
            Grid = make_grid(images, nrow=len(s2_vals))
            image = [torchvision.transforms.ToPILImage()(Grid)]
            #image = Grid
            #print("Grid complete.")

        if not return_dict:
            return image

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)

