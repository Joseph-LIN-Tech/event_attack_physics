import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
#from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader, 
    TexturesUV
)
import os.path as osp
#
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.src.human_body_prior.body_model.body_model import BodyModel
from PIL import Image
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj, save_obj

def load_data(data_name):
    amass_npz_fname = data_name # the path to body data
    bdata = np.load(amass_npz_fname)
    subject_gender = bdata['gender']
    subject_gender = str(subject_gender,encoding='utf-8')
    subject_gender=(bdata['gender'].tolist())
    return bdata,subject_gender


# # 载入SMPL模型
# def load_smpl_model_from_npz(npz_path, device):
#     subject_gender = r"neutral"
#     bdata,subject_gender = load_data(npz_path) 
#     bm_fname = osp.join('body_models/smpl_h_mano/{}/model.npz'.format(subject_gender))
#     dmpl_fname = osp.join('body_models/dmpls/{}/model.npz'.format(subject_gender))
#     num_betas = 16 # number of body parameters
#     num_dmpls = 8 # number of DMPL parameters
#     bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas,num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
#     faces = c2c(bm.f)
#     time_length = len(bdata['trans'])

#     body_parms = {
#         'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(device), # controls the global root orientation
#         'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(device), # controls the body
#         'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(device), # controls the finger articulation
#         'trans': torch.Tensor(bdata['trans']).to(device), # controls the global body position
#         'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device), # controls the body shape. Body shape is static
#         'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(device) # controls soft tissue dynamics
#     }
#     body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient', 'pose_hand','dmpls','betas']})#'root_orient','root_orient',
#     num_verts = bm.init_v_template.shape[1]
#     print("num_verts",num_verts)
#     verts = []
#     for i in range(time_length):
#         vertices=c2c(body_pose_beta.v[i])# vertices for each timestamp
#         #print(f"{i},vertices=",vertices)
#         verts.append(vertices)
#     return verts, faces 


def load_pose_from_file(pose_npz_path):
    subject_gender = r"neutral"
    #print("")
    bdata,subject_gender = load_data(pose_npz_path) 
    bm_fname = osp.join('body_models/smpl_h_mano/{}/model.npz'.format(subject_gender))
    dmpl_fname = osp.join('body_models/dmpls/{}/model.npz'.format(subject_gender))
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas,num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    faces = c2c(bm.f)
    #print("faces.shape",faces.shape)
    
    time_length = len(bdata['trans'])

    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]) # controls soft tissue dynamics
    }
    body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient', 'pose_hand','dmpls','betas']})#'root_orient','root_orient',
    #num_verts = bm.init_v_template.shape[1]
    #print("num_verts",num_verts)
    verts = []
    for i in range(time_length):
        vertices=c2c(body_pose_beta.v[i])# vertices for each timestamp
        #print(f"{i},vertices=",vertices)
        verts.append(vertices)
    return verts, faces 



# 渲染SMPL模型
def render_smpl_model(verts, faces, verts_uvs, faces_uvs, texture_image,H,W,device,R1=-1,R2=1,R3=1,T_val=1.75):
    textures = TexturesUV(maps=texture_image, faces_uvs=[faces_uvs], verts_uvs=[verts_uvs])
    smpl_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    #print("in render smpl ", R1, R2, R3, "T_val", T_val)
    R =  torch.tensor([[0,0,R1],[R2,0,0],[0, R3, 0]],device=device)
    
    T = torch.tensor([0,0,T_val],device=device)
    cameras = PerspectiveCameras(device=device, R=R[None], T=T[None])

    raster_settings = RasterizationSettings(image_size=(H,W), blur_radius=0.0, bin_size = 0, faces_per_pixel=10)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    images = renderer(smpl_mesh).to(device)
    return images[0, ..., :3]

# 优化UV贴图
def optimize_uv_texture(verts, faces, verts_uvs, faces_uvs, uv_texture_path, device):
    faces = torch.tensor(faces, dtype=torch.long, device=device)#.unsqueeze(0)
    uv_texture = Image.open(uv_texture_path).convert("RGB") 
    uv_texture = np.array(uv_texture) / 255.0
    uv_texture = torch.tensor(uv_texture, dtype=torch.float32).unsqueeze(0).to(device)
    #print("uv_texture.shape",uv_texture.shape)
    uv_texture.requires_grad = True
    optimizer = optim.Adam([uv_texture], lr=0.01)

    target_class = 0  # 假设目标是攻击YOLO检测到的行人类别
    print("len(verts)",len(verts))
    for epoch in range(len(verts)):
        optimizer.zero_grad()
        verts_sub = torch.tensor(verts[epoch],  device=device).squeeze(0)    
        rendered_image = render_smpl_model(verts_sub, faces, verts_uvs, faces_uvs, uv_texture,device)*255.0
        #print("rendered_image.shape",rendered_image.shape)
        image_array = rendered_image.detach().cpu().numpy().astype(np.uint8)  # 将浮点数转换为8位无符号整数
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(image_array)
        # 保存为PNG文件
        #output_path = f'/home/lgx/code/ECCV2024/white_box_attack/attack_letgan_new_uv/output_data/output_image_{epoch}.png'
        #image.save(output_path)
    return uv_texture#.detach()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_params_path = '/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_cell_part/support_data/cmu_data/02_01_poses.npz'  # 你的SMPL模型文件路径
    #uv_texture_path = '/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/smpl_uv.png'  # 你的UV贴图文件路径
    uv_texture_path = "/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/patch1.png"
    uv_obj_path = "/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/smpl_uv.obj"
    smpl_model_path = r"/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data"
    verts, faces, aux = load_obj(
        uv_obj_path,
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=8,
        texture_wrap=None,
    )

    verts_uvs = aux.verts_uvs.to(device)
    faces_uvs = faces.textures_idx.to(device)

    verts, faces= load_smpl_model_from_npz(smpl_params_path,smpl_model_path,device)
    optimized_uv_texture = optimize_uv_texture(verts, faces, verts_uvs, faces_uvs, uv_texture_path, device)

if __name__ == "__main__":
    main()
