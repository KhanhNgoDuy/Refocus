import sys
sys.path.append('..')
from typing import Any, Callable, Dict, List, Optional, Union
import itertools
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# from models.camera_model import CLIPCameraProjection


class MyPipeline(DiffusionPipeline):
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
    
    def get_params_to_optimize(self):
        return itertools.chain(
            self.unet.parameters(),
            self.camera_projection.parameters()
        )

    def train_modules(self):
        self.unet.train()
        self.camera_projection.train()

    def eval_modules(self):
        self.unet.eval()
        self.camera_projection.eval()
    
    def config_modules(self):
        expected = next(iter(self.vae.parameters()))
        self.camera_projection.to(expected)

    def encode_vae(self, image_pt: torch.FloatTensor, num_images_per_prompt: int = None):
        image_pt = image_pt.to(self.vae.device, self.vae.dtype)
        image_pt = image_pt * 2.0 - 1.0  # scale to [-1, 1]
        image_pt = self.vae.encode(image_pt).latent_dist.sample()
        image_pt = image_pt * self.vae.config.scaling_factor      # khanh: test this later --> worked
        if num_images_per_prompt is not None:
            image_pt = image_pt.repeat_interleave(num_images_per_prompt, dim=0)

        return image_pt
    
    def encode_cam(self, cam):
        expected = next(iter(self.camera_projection.parameters()))
        cam = cam.to(expected)
        return self.camera_projection(cam)

    def train_step(self, tgt, src, masked, depth, cam):
        self.config_modules()
        
        # khanh: encode conditional image (masked) & camera pose
        image_latents = self.encode_vae(src)
        masked_latents = self.encode_vae(masked)
        encoder_hidden_states = self.encode_cam(cam).unsqueeze(0)
        
        # khanh: encode input image (original)
        latents = self.encode_vae(tgt)
        
        # khanh: sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        latent_model_input = torch.cat([noisy_latents, masked_latents, image_latents], dim=1)
        noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample
        
        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        
        return loss
        
    
    @torch.no_grad()
    def __call__(
        self,
        src: torch.FloatTensor,
        masked: torch.FloatTensor,
        depth: torch.FloatTensor,
        cam: torch.FloatTensor,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        use_progress_bar=True
    ):
        self.config_modules()
        device = self._execution_device

        # 3. Encode inputs
        image_latents = self.encode_vae(src)
        masked_latents = self.encode_vae(masked)
        encoder_hidden_states = self.encode_cam(cam).unsqueeze(0)
        
        # khanh: prepare random noise
        latents = randn_tensor(masked_latents.shape, generator=generator, device=device, dtype=encoder_hidden_states.dtype)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if not use_progress_bar:
            self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents, masked_latents, image_latents], dim=1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, False)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=False
        )
        
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [
            self.unet,
            self.image_encoder,
            self.vae,
            self.safety_checker,
        ]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property    
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device



def build_model(rank):
    # pipe = MyPipeline.from_pretrained("ashawkey/stable-zero123-diffusers", torch_dtype=torch.float32, trust_remote_code=True)
    pipe = MyPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float32)
    pipe.vae.requires_grad_(False)
    
    # duplicate the channel of unet by 3
    conv = pipe.unet.conv_in
    conv_dup = nn.Conv2d(conv.in_channels * 3, conv.out_channels, conv.kernel_size, conv.stride, conv.padding)
    weight = conv.weight.repeat(1, 3, 1, 1) / 3
    conv_dup.weight.data = weight
    conv_dup.bias.data = conv.bias.data
    pipe.unet.conv_in = conv_dup
    
    # clip-camera-projection
    pipe.camera_projection = nn.Linear(2, 1024)
    
    pipe.to(f"cuda:{rank}")
    
    return pipe


if __name__ == "__main__":
    pipe = build_model(0)
    
    bsz=1
    image = torch.randn(bsz, 3, 256, 256).to(dtype=torch.float32).clamp(min=0, max=1)
    blurred = torch.randn_like(image).to(image).clamp(min=0, max=1)
    masked = torch.randn_like(image).to(image).clamp(min=0, max=1)
    depth = torch.randn_like(image).to(image).clamp(min=0, max=1)
    cam = torch.rand(1, 2).to(image)
        
    loss = pipe.train_step(blurred, image, masked, depth, cam)
    print(loss)
    noise = pipe(image, masked, depth, cam)['images'][0]
    print(noise.size)
