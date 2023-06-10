import os
import argparse
import logging
import sys

from diffusers import StableDiffusionPipeline
import torch

#####
from subprocess import run

from safetensors.torch import load_file
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector

from diffusers.models import AutoencoderKL
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel
from controlnet_aux import OpenposeDetector

import gc
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from diffusers.models import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel
import os
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

###

def load_lora_weights(pipeline, checkpoint_path, lora_alpha):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = lora_alpha
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

model_id = os.environ.get("TRAINML_CHECKPOINT_PATH")
print (f"MODEL_ID path is: {model_id}")
data = run(f"ls -al {model_id}",capture_output=True,shell=True)
print(data.stdout)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a custom model prompt generation.")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A photo of sks dog in jumping over the moon",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="",
        help="negative prompt"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="number of images",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default="nouser",
        help="user id",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="nouser",
        help="token",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    args = parse_args()
    #if args.seed != -100:
#   generator = torch.Generator("cuda").manual_seed(args.seed)
    #else:
    #    generator = torch.Generator("cuda")
    # negative_prompt=negative_pr,
    checkpoint_version = "checkpoint-250"
    sec_checkpoint_version = model_id

    lora_model_path = "models/more_details.safetensors"
    use_lora = True
    lora_alpha = 0.75

    ### OPENPOSE
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    image = load_image(
        "data/pose/output_8.png"
    )
    openpose_image = openpose(image)
    ##############
    logging.info('Loaded openpose')
    ### CONTROLNET
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    logging.info('Loaded controlnet')
    ### VAE
    vae_to_use = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    logging.info('Loaded vae')
    ### LOAD UNET AND ENCODER
    unet_model = UNet2DConditionModel.from_pretrained(f"{model_id}/{checkpoint_version}/", subfolder="unet", torch_dtype=torch.float16)
    text_enc = CLIPTextModel.from_pretrained(f"{model_id}/{checkpoint_version}/", subfolder="text_encoder", torch_dtype=torch.float16)

    logging.info('Loaded vae, unet, encoder')


    ### Main pipe
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        vae=vae_to_use,
        text_encoder = text_enc,
        unet=unet_model
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    logging.info('Loaded main pipeline')
    ### IF LORA IS SET TO LOAD, THEN LOAD
    if use_lora == True:
        pipe = load_lora_weights(pipe, lora_model_path, lora_alpha)
    else:
        pipe.to("cuda")

    #pipe.enable_model_cpu_offload()

    ### DISA
    #def dummy(images, **kwargs):
    #    return images, False
    #pipe.safety_checker = dummy

    ### PROMPTS


    prompt1 =f"<lora:more_details:1.5> redshift style, painted portrait of {args.token} a paladin, masculine, mature, handsome, grey and silver, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, gaston bussiere, alphonse mucha"
    prompt2 =f"<lora:more_details:1.5> stain glass window of {args.token} as god warrior, light shiny through, intricate, elegant, highly detailed, digital painting, sharp focus, realistic, hyperrealistic, cinematic, illustration"
    prompt3 =f"<lora:more_details:1.5> A man made of fire, intricate heat distortion designs, elegant, highly detailed, sharp focus, art by Artgerm and Greg Rutkowski and WLOP,{args.token}"
    prompt4 =f"<lora:more_details:1.5> neon light sign in design of face|{args.token} as gorgeous god | detailed gorgeous face | precise lineart | intricate | rea listic | studio quality | cinematic | luminescence | character design | concept art | highly detailed | illustration | digital art | digital paintin"
    prompt5 =f"<lora:more_details:1.5> {args.token},poster of warrior god, standing alone on hill, centered, detailed gorgeous face, anime style, key visual, intricate detail, highly detailed, breathtaking, vibrant, panoramic, cinematic, Carne Griffiths, Conrad Roset, Makoto Shinkai"
    prompt6 =f"<lora:more_details:1.5> portrait of {args.token} as a rugged 19th century man with mutton chops in a jacket, victorian, concept art, detailed face, fantasy, close up face, highly detailed, cinematic lighting, digital art painting by (greg rutkowski)"
    prompt7 =f"<lora:more_details:1.5> {args.token}, ((tarot card with intricate detailed frame around the outside)) | side profile of cyberpunk head with large moon in background| cyberpunk | styled in Art Nouveau | insanely detailed | embellishments | high definition | concept art | digital art | vibrant"
    prompt8 =f"<lora:more_details:1.5> {args.token},comic book panel, lineart illustration , sharp eyes focus , redhead,((man in plate armor )), ((ultra realistic detailed eyes)), intricate eye details, cute smile, sweaty , ultra skin texture, ((ultra skin details)),intricate details, (in dark sci-fi space ship)"


    prompt_arr = [prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7, prompt8]

    neg_p = """(deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, 
    low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, 
    dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,
    deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"""

    seed=args.seed
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        prompt_arr,
        [openpose_image] * 8,
        negative_prompt=[neg_p] * 8,
        num_inference_steps=50,
        guidance_scale = 7.5,
        controlnet_conditioning_scale =0.4,
        #generator=generator
    )

    ### LOAD SECONDARY UNET AND ENCODER 
    u_unet_model = UNet2DConditionModel.from_pretrained(f"{model_id}", subfolder="unet", torch_dtype=torch.float16)
    u_text_enc = CLIPTextModel.from_pretrained(f"{model_id}", subfolder="text_encoder", torch_dtype=torch.float16)

    i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler"),
        torch_dtype=torch.float16,
        vae=vae_to_use,
        unet=u_unet_model,
        text_encoder=u_text_enc
    )

    if use_lora == True:
        i2i_pipe = load_lora_weights(i2i_pipe, lora_model_path, lora_alpha)
    else:
        i2i_pipe.to("cuda")

    i=0
    all_images = [] 
    for i in range(len(output.images)):
        #print (prompt_arr[i])
        i2i_images = i2i_pipe(prompt=prompt_arr[i], negative_prompt=neg_p, num_images_per_prompt=1, image=output.images[i], strength=0.5, guidance_scale=7.5).images
        all_images.extend(i2i_images)
        all_images.extend(output.images[i])
    #images = pipe(args.prompt, negative_prompt = args.negative_prompt, num_images_per_prompt=args.num, num_inference_steps=args.steps, guidance_scale=args.scale).images
    #all_images.append(output.images)
    i=0
    print ("I GOT TO HERE BEFORE SAVING")
    for img in all_images:
        img.save(f"{os.environ.get('TRAINML_OUTPUT_PATH')}/output_{i}.png")
        i = i+1
