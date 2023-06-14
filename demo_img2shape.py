import os
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif
from utils.demo_util import SDFusionImage2ShapeOpt
from pytorch3d.io import load_objs_as_meshes, save_obj

cudnn.benchmark = True
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
seed = 2023
opt = SDFusionImage2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device
vq_ckpt_path='saved_ckpt/shapenet_img2shape/vqvae_steps-latest.pth'
ckpt_path = 'saved_ckpt/shapenet_img2shape/df_steps-latest.pth'
img_dir = 'demo_imgs/shapenet_img2shape/'
opt.init_model_args(ckpt_path=ckpt_path, vq_ckpt_path=vq_ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')
from utils.demo_util import preprocess_image
# img2shape
out_dir = 'demo_results/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


ngen = 1 # number of generated shapes
ddim_steps = 100
ddim_eta = 0.
uc_scale = 3.
input_img = os.path.join(img_dir, "03001627_1de49c5853d04e863c8d0fdfb1cc2535_00003.white.png")
sdf_gen = SDFusion.img2shape(image=input_img, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)
mesh_gen = sdf_to_mesh(sdf_gen)

verts, faces = mesh_gen.get_mesh_verts_faces(0)
mesh_path = os.path.join(out_dir, f'demo.obj')
save_obj(mesh_path, verts, faces)
# vis as gif

gen_name = f'{out_dir}/demo.gif'

save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)