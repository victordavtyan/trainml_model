import os
import argparse
import logging
import sys

from diffusers import StableDiffusionPipeline
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

model_id = os.environ.get("TRAINML_CHECKPOINT_PATH")
print ("MODEL SAVE PATH IS:")
print(os.environ.get('TRAINML_OUTPUT_PATH'))
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")


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
        "--steps",
        type=int,
        default=50,
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
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    args = parse_args()
    negative_pr = "((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), (((trans))), (hermaphrodite), ((out of frame)), ((extra fingers)), ((mutated hands)), ((poorly drawn hands)), ((poorly drawn face)), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((scary face))"
    images = pipe(args.prompt, negative_prompt=negative_pr, num_images_per_prompt=args.num, num_inference_steps=args.steps, guidance_scale=args.scale).images

    for i, img in enumerate(images):
        img.save(f"{os.environ.get('TRAINML_OUTPUT_PATH')}/output_{i}.png")
