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
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler
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
import requests

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

def dummy(images, **kwargs):
        return images, len(images)*[False]

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
        default=-1,
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
        "--gender",
        type=str,
        default="women",
        help="gender",
    )
    parser.add_argument('--lora', default=False, action='store_true')
    
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0
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
    checkpoint_version = "checkpoint-350"
    #checkpoint_version = model_id
    sec_checkpoint_version = model_id

    lora_model_path = "models/more_details.safetensors"
    use_lora = args.lora
    lora_alpha = args.lora_alpha

    if args.gender == "women":
        gender = "female"
    else:
        gender = "male"

    num_per_prompt = 4

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

    #unet_model = UNet2DConditionModel.from_pretrained(f"{model_id}", subfolder="unet", torch_dtype=torch.float16)
    #text_enc = CLIPTextModel.from_pretrained(f"{model_id}", subfolder="text_encoder", torch_dtype=torch.float16)


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
    
    #pipe.load_textual_inversion("models/FastNegativeV2.pt")
    logging.info('Loaded textual inversion') # UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    logging.info('Loaded main pipeline')
    ### IF LORA IS SET TO LOAD, THEN LOAD
    if use_lora == True:
        pipe = load_lora_weights(pipe, lora_model_path, lora_alpha)
    else:
        pipe.to("cuda")

    #pipe.enable_model_cpu_offload()

    ### DISA
    
    #pipe.safety_checker = dummy

    ### PROMPTS

    api_url = 'https://imaginate.rebit.ai/api/styles-with-prompts'
    r_params = dict()
    resp = requests.get(url=api_url, params=r_params)
    data = resp.json() # Check the JSON Response Content documentation below

    
    prompt_arr = []
    prompt_ids = []
    style_ids = []
    neg_p_arr = []
    for style in data['success']['styles']:
        if style[f"for_{gender}"] == 1:
            
            for prompt in style['prompts']:
                prompt_arr.append(prompt["text"].replace("{args.token}", f"{args.token}"))
                #prompt_arr.append(prompt["text"].replace("{args.token}", f"{args.token}"))
                prompt_ids.append(prompt["id"])
                neg_p_arr.append(prompt["negative"])
                style_ids.append(prompt["style_id"])

    #prompt1 =f"redshift style, painted portrait of {args.token} a paladin,  masculine, mature, handsome, grey and silver, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, gaston bussiere, alphonse mucha"
    #prompt2 =f"stain glass window of {args.token} as god warrior,  light shiny through, intricate, elegant, highly detailed, digital painting, sharp focus, realistic, hyperrealistic, cinematic, illustration"
    #prompt3 =f"A man made of fire, intricate heat distortion designs,  elegant, highly detailed, sharp focus, art by Artgerm and Greg Rutkowski and WLOP,{args.token}"
    #prompt1 = f"redshift style, painted portrait of {args.token} a paladin,colorfull,masculine, mature, handsome,silver,gold and blue, fantasy armor, intricate, elegant, highly detailed, digital painting,artstation, concept art, smooth, sharp focus, illustration, gaston bussiere, alphonse mucha"
    #prompt2 = f"Portrait of {args.token}, charliebo artstyle, viking warrior,medieval armor, fantasy,elegant,sharp eyes focus,handsome,epic composition,highly detailed, intricate, digital painting, trending on artstation, concept art, smooth, dark, gloomy, realistic, illustration, 8k, 4k, dramatic lighting, d&d"
    #prompt3 = f"{args.token} as a powerful mysterious wizard, casting lightning magic,(blue lightning,flash),detailed clothing, digital painting, fantasy, Surrealist, by Stanley Artgerm Lau and Alphonse Mucha, artstation, highly detailed, sharp focus, stunningly beautiful, dystopian, iridescent gold, cinematic lighting, dark"
    
    #prompt4 =f"neon light sign in design of face|{args.token} as gorgeous god | detailed gorgeous face | precise lineart | intricate | rea listic | studio quality | cinematic | luminescence | character design | concept art | highly detailed | illustration | digital art | digital paintin"
    #prompt5 =f"{args.token},poster of warrior god, standing alone on hill,  centered, detailed gorgeous face, anime style, key visual, intricate detail, highly detailed, breathtaking, vibrant, panoramic, cinematic, Carne Griffiths, Conrad Roset, Makoto Shinkai"
    #prompt6 =f"portrait of {args.token} as a rugged 19th century man with mutton chops in a jacket,  victorian, concept art, detailed face, fantasy, close up face, highly detailed, cinematic lighting, digital art painting by (greg rutkowski)"
    #prompt1=f"redshift style, painted portrait of {args.token} a paladin,colorfull,masculine, mature, handsome,silver,gold and blue, fantasy armor, intricate, elegant, highly detailed, digital painting,artstation, concept art, smooth, sharp focus, illustration, gaston bussiere, alphonse mucha"
    #prompt2=f"Portrait of {args.token}, charliebo artstyle, viking warrior,medieval armor, fantasy,elegant,sharp eyes focus,handsome,epic composition,highly detailed, intricate, digital painting, trending on artstation, concept art, smooth, dark, gloomy, realistic, illustration, 8k, 4k, dramatic lighting, d&d"
    #prompt3=f"{args.token} as a powerful mysterious wizard, casting lightning magic,(blue lightning,flash),detailed clothing, digital painting, fantasy, Surrealist, by Stanley Artgerm Lau and Alphonse Mucha, artstation, highly detailed, sharp focus, stunningly beautiful, dystopian, iridescent gold, cinematic lighting, dark"
    #prompt4=f"beautiful satanic male,necromancer blood king,portrait of {args.token},ritualistic,oil painting, angular features,magician cloak,in style of Greg Rutkowski,blood drops,red and black, illustration, 8k, 4k,dramatic lighting,((dark and gloomy universe)),smoke and shadow, night"
    #prompt5=f"Digital painting of a beautiful man android,{args.token}, cyborg, robotic parts,(sci-fi costume),camo battle armor, studio soft light, rim light, vibrant details, luxurious cyberpunk, (lace), anatomical, facial muscles,ultra detailed, cable electric wires, microchip, elegant, beautiful background, octane render,style of Stephen Hickman"
    #prompt6=f"poster of a god, portrait of {args.token} with hyperdimensional totem implants,neon arnaments, bio luminescent, water, plasma, creature,turquoise and violet, artwork by tooth wu and wlop and android jones and greg rutkowski"
    #prompt7 =f"{args.token}, ((tarot card with intricate detailed frame around the outside)) | side profile of cyberpunk head with large moon in background| cyberpunk | styled in Art Nouveau | insanely detailed | embellishments | high definition | concept art | digital art | vibrant"
    #prompt8 =f"{args.token}, comic book panel, lineart illustration ,  sharp eyes focus , redhead,((man in plate armor )), ((ultra realistic detailed eyes)), intricate eye details, cute smile, sweaty , ultra skin texture, ((ultra skin details)),intricate details, (in dark sci-fi space ship)"


    #prompt_arr = [prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7, prompt8]
    #print(prompt_ids)
    #print(style_ids)
    #neg_p = """(FastNegativeV2:0.5), (deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, 
    #low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, 
    #dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,
    #deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"""
    #neg_p1="necktie,tie,(deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"
    #neg_p2="tie,horn,(deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"
    #neg_p3="tie,suit,(deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"
    #neg_p4="frame,suit,horns(deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"
    #neg_p5="frame,suit,mask,(deformed iris, deformed pupils :1.4),text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"
    #neg_p6="text,frame,suit,(deformed iris, deformed pupils :1.4), cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,cloned face, disfigured, gross proportions, malformed limbs,long neck,deformed skin,(robot eyes, bad eyes, crosseyed, small eyes:1.3)"
    
    #neg_p_arr = [neg_p1,neg_p2,neg_p3,neg_p4,neg_p5,neg_p6,neg_p,neg_p]

    seed=args.seed
    generator = torch.Generator("cuda").manual_seed(seed)
    output_images = []

    n = 5 
    prompt_chunks = [prompt_arr[i * n:(i + 1) * n] for i in range((len(prompt_arr) + n - 1) // n )] 
    negp_chunks = [neg_p_arr[i * n:(i + 1) * n] for i in range((len(neg_p_arr) + n - 1) // n )] 

    i=0
    for prompt_chunk in prompt_chunks:

        output = pipe(
            prompt_chunk,
            [openpose_image] * len(prompt_chunk),
            negative_prompt=negp_chunks[i],
            num_images_per_prompt=num_per_prompt,
            num_inference_steps=50,
            guidance_scale = 7.5,
            controlnet_conditioning_scale=0.4,
        #    generator=generator
        )
        i=i+1
        output_images.extend(output.images)
        torch.cuda.empty_cache()

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
    
    #i2i_pipe.load_textual_inversion("models/FastNegativeV2.pt")

    if use_lora == True:
        i2i_pipe = load_lora_weights(i2i_pipe, lora_model_path, lora_alpha)
    else:
        i2i_pipe.to("cuda")

    #i2i_pipe.safety_checker = dummy

    i2i_generator = torch.Generator("cuda").manual_seed(seed)
    i=0
    all_images = [] 
    prompt_counter = 0
    for i in range(len(output_images)):
        #print (prompt_arr[i])
        i2i_images = i2i_pipe(prompt=prompt_arr[prompt_counter], negative_prompt=neg_p_arr[prompt_counter], num_images_per_prompt=1, image=output_images[i], strength=0.5, guidance_scale=7.5).images
        all_images.extend(i2i_images)

        if (i+1) % num_per_prompt == 0:
            prompt_counter = prompt_counter + 1        
        
    #images = pipe(args.prompt, negative_prompt = args.negative_prompt, num_images_per_prompt=args.num, num_inference_steps=args.steps, guidance_scale=args.scale).images , generator=i2i_generator
    #all_images.append(output.images)
    #all_images.extend(output.images)
    i=0
    prompt_counter = 0
    
    for img in all_images:

        prompt_id = prompt_ids[prompt_counter]
        style_id = style_ids[prompt_counter]
          
        img.save(f"{os.environ.get('TRAINML_OUTPUT_PATH')}/output_{style_id}_{prompt_id}_{i}.png")
    
        if (i+1) % num_per_prompt == 0:
            prompt_counter = prompt_counter + 1    
        
        i = i+1