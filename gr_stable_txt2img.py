# python3 gr_stable_txt2img.py --ckpt logs/dog2022-11-13T05-23-51_dog/checkpoints/last.ckpt --token "mmdd111"

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
#from torch import autocast
from torch.cuda.amp import autocast as autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import gradio as gr
import PIL

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    #print("\n".join(list(filter(lambda x: x.find("model.")==0 and x.find("attn")>0 ,sd.keys()))))
    #sys.exit(0)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def main(prompt, t0, guidance_scale, nb_steps, nsfw_filter):
    print("token: {}, edit prompt: {}".format(opt.token, prompt))

    outpath = opt.outdir

    batch_size = 1

    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        #with precision_scope("cuda"):
        with precision_scope():
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                prompts = batch_size * [opt.token+" "+prompt]
                uc = None
                if guidance_scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                c0 = model.get_learned_conditioning(batch_size * [opt.token])
                cedit = model.get_learned_conditioning(batch_size * [prompt])

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=nb_steps,
                                                 conditioning=c,
                                                 cedit=cedit,
                                                 conditioning0=c0,
                                                 t0=t0,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=start_code)

                #x_samples_ddim = model.decode_first_stage(samples_ddim)
                image = model.decode_first_stage(samples_ddim)

                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = numpy_to_pil(image)

                print(f"Your samples are ready and waiting for you here: \n{outpath}\n\nEnjoy.")

                return {"sample": image, "nsfw_content_detected": False} 

def run(
    *,
    prompt,
    nb_steps,
    t0,
    guidance_scale,
    pipe,
):

    #with autocast("cuda"):
    with autocast():
        images = pipe(
            prompt,
            t0=t0,
            guidance_scale=guidance_scale,
            nb_steps=nb_steps,
            nsfw_filter=True,
        )["sample"]

        return images[0]


def gradio_run(
    prompt,
    nb_images=1,
    nb_steps=50,
    t0=1.,
    guidance_scale=7.,
):

    images = []

    for _ in range(nb_images):
        generated = run(
            prompt=prompt,
            nb_steps=nb_steps,
            t0=t0,
            guidance_scale=guidance_scale,
            pipe=main,
        )

        images.append(generated)

    return images


inpaint_interface = gr.Interface(
    gradio_run,
    inputs=[
        gr.Textbox(),
        gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Number of images"),
        gr.Slider(minimum=1, maximum=200, value=50, label="Number of steps"),
        gr.Slider(minimum=0, maximum=1, value=1., label="T0"),
        gr.Slider(minimum=0, maximum=20, value=7.0, label="Guidance scale"),
    ],
    outputs=[
        gr.Gallery(label="Images"),
    ],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--token",
        type=str,
        nargs="?",
        default="mmdd111",
        help="the token to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--blendmodel",
        action='store_true',
        default=False,
        help="blend with original model, check ldm/models/diffusion/ddim.py",
    )
    parser.add_argument(
        "--blendpos",
        type=int,
        default=-1,
        help="blend with original model, check ldm/models/diffusion/ddim.py",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )


    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    #model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    blendmodel = None

    if opt.plms:
        sampler = PLMSSampler(model, blendmodel, opt.blendpos)
    else:
        sampler = DDIMSampler(model, blendmodel, opt.blendpos)

    os.makedirs(opt.outdir, exist_ok=True)

    inpaint_interface.launch(server_name="0.0.0.0", server_port=8032)


# https://github.com/leszekhanusz/diffusers/blob/feature_unified_stable_diffusion_pipeline/examples/inference/unified_gradio.py
