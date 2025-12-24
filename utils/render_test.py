import glob
from attack_event_NMI_fullmask.utils.call_yolo import *
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
from pytorch3d.io import load_obj, save_obj

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
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from PIL import Image
import matplotlib.pyplot as plt

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return [int(num) if num.isdigit() else num for num in numbers]

def find_pattern_score(str1,pattern):
    #box_pattern = r"\[\[.*?\]\]"
    matches = re.findall(pattern, str1)
    print(matches)
    if matches:
        numbers_str = matches[0]
        numbers_str = numbers_str.replace("[","").replace("]","").split(' ')[1]#[-1]
        #print("numbers_str",numbers_str)

        score = float(numbers_str)#np.fromstring(numbers_str, sep=' ').astype('float')
        #print("score")
        print(score)
    else:
        print("No match found.")
        
    return score

def load_data(data_name):
    amass_npz_fname = data_name # the path to body data
    bdata = np.load(amass_npz_fname)
    subject_gender = bdata['gender']
    subject_gender = str(subject_gender,encoding='utf-8')
    subject_gender=(bdata['gender'].tolist())
    return bdata,subject_gender

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
def render_smpl_model(verts, faces, verts_uvs, faces_uvs, texture_image,H,W,device):
    textures = TexturesUV(maps=texture_image, faces_uvs=[faces_uvs], verts_uvs=[verts_uvs])
    smpl_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    R =  torch.tensor([[0,0,-1],[1,0,0],[0, 1, 0]],device=device)
    T = torch.tensor([0,0,2],device=device)
    cameras = PerspectiveCameras(device=device, R=R[None], T=T[None])

    raster_settings = RasterizationSettings(image_size=(H,W), blur_radius=0.0, bin_size = 0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    images = renderer(smpl_mesh).to(device)
    return images[0, ..., :3]

def transform_image(image,target_size):
    h, w, _ = image.shape
    target_h, target_w = target_size
    #original_height, original_width = image.shape[:2]

    # Calculate the new size and scale for the image
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a new image with the target size and fill it with a gray color
    padded_image = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Place the resized image in the center of the padded image
    padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = resized_image

    return padded_image, scale, pad_x, pad_y

def map_back_to_original(x,y, scale, pad_x, pad_y):
    x = (x - pad_x) / scale
    y = (y - pad_y) / scale        
    return x,y


def preprocess(input_images_path, yolox_path,output_npz_path, output_labeled_img_path,scale,pad_x, pad_y, dataset_resolution_format='gen1'):
    '''
    Purpose: to create the label of human for the detection
    '''
    video_list = os.listdir(input_images_path)
    for video in video_list:
        video_name = str(video)
        video_path = os.path.join(input_images_path,video)
        output_npz_path = os.path.join(output_npz_path,video_name)
        if not os.path.exists(output_npz_path):
            os.makedirs(output_npz_path)

        output_labeled_img_path = os.path.join(output_labeled_img_path,video_name)
        if not os.path.exists(output_labeled_img_path):
            os.makedirs(output_labeled_img_path)

        pose_files = os.listdir(video_path)
        track_id = 0
        for pose_file in pose_files:
            pose_name = pose_file.split("/")[-1]
            pose_path = os.path.join(video_path,pose_name)
            rgb_img_path = glob.glob(os.path.join(pose_path,"*.png"))
            output_labeled_img_file = os.path.join(output_labeled_img_path,pose_file)
            if not os.path.exists(output_labeled_img_file):
                os.makedirs(output_labeled_img_file) 
            rgb_img_path_sorted = sorted(rgb_img_path, key=numerical_sort)
            array_list = []
            
            track_id += 1
            fps = 120
            time_interval_ms = 1000/fps # ms
            index = 0
            for img_path in rgb_img_path_sorted:
                t = (int)(index*time_interval_ms*1000)  #   us 
                return_val =callback_onnx_yolox_s(yolox_path, img_path,output_labeled_img_file,cuda_id="1")

                box_pattern = [r"\[\[.*?\]\]",r"\[\[.*?\]\n"]
                box_coord = find_pattern_bbox(return_val,box_pattern)

                if dataset_resolution_format == "gen1":
                    x_lefttop = box_coord[0][0]   # map [0,640] into [0,320]
                    y_lefttop = box_coord[0][1]   # map [0,640] into [0,320]
                    x_lefttop,y_lefttop = map_back_to_original(x_lefttop,y_lefttop ,scale, pad_x, pad_y)

                    x_rightbot = box_coord[0][2]  # map [0,640] into [0,320]
                    y_rightbot = box_coord[0][3]  # map [0,640] into [0,320]
                    x_rightbot,y_rightbot = map_back_to_original(x_rightbot,y_rightbot ,scale, pad_x, pad_y)

                w = x_rightbot - x_lefttop
                h = y_rightbot - y_lefttop
                assert (w>0 and h>0), print("h and w must larger than 0")
                class_id = 0  # car is 0, human is 1, to make the attack sucessful, we need to set the GT label as car
                #track_id = id
                score_pattern = r"final\_scores \[.*?\]"
                confidence_score = find_pattern_score(return_val,score_pattern)
                
                # if confidence_score > 0.6:
                #     pass 
                # else:
                #     raise("confidence_score is low")

                array = [(int)(t),(int)(x_lefttop),(int)(y_lefttop),(int)(w),(int)(h),(int)(class_id),confidence_score,track_id]
                print("array",array)
                array_list.append(array)
                index+=1
                # if index == 10:
                #     break

            # 保存NumPy数组为.npy文件
            array_path = os.path.join(output_npz_path,f'{pose_name}.npz')
            np.savez(array_path, labels=array_list)

            sequence_labels = np.load(str(array_path))
            print("sequence_labels",sequence_labels['labels'])
            #break
def render_images(verts, faces, verts_uvs, faces_uvs, uv_texture, output_path, H,W, target_size, device):
    faces = torch.tensor(faces, dtype=torch.long, device=device)#.unsqueeze(0)
    #uv_texture = Image.open(uv_texture_path).convert("RGB") 
    #uv_texture = np.array(uv_texture) / 255.0
    #uv_texture = torch.tensor(uv_texture, dtype=torch.float32).unsqueeze(0).to(device)
    for i in range(len(verts)):
        verts_sub = torch.tensor(verts[i],  device=device).squeeze(0)    
        rendered_image = render_smpl_model(verts_sub, faces, verts_uvs, faces_uvs, uv_texture,H,W,device)*255.0
        image_array = rendered_image.detach().cpu().numpy().astype(np.uint8)  # 将浮点数转换为8位无符号整数
        print("before, image_array.shape",image_array.shape)
        image_array, scale, pad_x, pad_y = transform_image(image_array,target_size)# H W 240X304 -> 640 x 640
        print("after, image_array.shape: ",image_array.shape," scale: ",scale, " padding_x: ",pad_x, "padding_y: ",pad_y)
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(image_array)
      
        output_img_name = os.path.join(output_path,f"{i:05d}.png")
        print("save img: ",output_img_name)
        image.save(output_img_name)

def load_pose_and_rendering(smpl_params_path,verts_uvs,faces_uvs,output_path,H,W,target_size,original_uv_img_path, device):
    
    video_list = os.listdir(smpl_params_path)
    uv_texture = cv2.imread(original_uv_img_path).astype(np.float32)/255.
    uv_texture = uv_texture[..., [2, 1, 0]] #BGR -> RGB#torch.ones((1,512,512,3),device=device)
    uv_texture = torch.from_numpy(np.ascontiguousarray(uv_texture)).unsqueeze(0)
    uv_texture = uv_texture.to(device=device)
    print("uv_texture.shape",uv_texture.shape)
    for video in video_list:
        video_name = str(video)
        video_path = os.path.join(smpl_params_path,video)
        pose_files = glob.glob(os.path.join(video_path,"*.npz"))
        output_video_path = os.path.join(output_path,video_name)
        if not os.path.exists(output_video_path):
            os.makedirs(output_video_path)

        for pose_file in pose_files:
            pose_name = pose_file.split("/")[-1][:-4]
            verts, faces = load_pose_from_file(pose_file)
            output_pose_image_path = os.path.join(output_video_path,pose_name)
            if not os.path.exists(output_pose_image_path):
                os.makedirs(output_pose_image_path)
            render_images(verts, faces, verts_uvs, faces_uvs,uv_texture, output_pose_image_path,H,W, target_size,device)
   


import torch

def main():
    render_images = True
    labeling_images = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolox_path = "/home/lgx/code/NMI/attack_event_NMI/checkpoints/YOLOX"
    smpl_params_path = '/home/lgx/data/whitebox_attack_cmupose/train'  # 你的SMPL模型文件路径
    #uv_texture_path = '/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/smpl_uv.png'  # 你的UV贴图文件路径
    #uv_texture_path = "/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/patch1.png"
    uv_obj_path = "/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/smpl_uv.obj"
    smpl_model_path = r"/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data"
    output_rendered_images = "/home/lgx/data/whitebox_attack_cmupose/train_rendered_images_test"
    original_uv_img_path = "/home/lgx/code/NMI/attack_event_NMI/Data/smpl_uv2.png"
    H, W= 240,304
    target_size = (640,640)
    target_h = target_size[0]
    target_w = target_size[1]
    scale = min(target_w / W, target_h / H)
    new_w = int(W * scale)
    new_h = int(H * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    if render_images:
        if not os.path.exists(output_rendered_images):
            os.makedirs(output_rendered_images)

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
        load_pose_and_rendering(smpl_params_path,verts_uvs,faces_uvs,output_rendered_images, H,W,target_size, original_uv_img_path, device)

    if labeling_images:
        input_images_path = "/home/lgx/data/whitebox_attack_cmupose/train_rendered_images"
        output_npz_path = "/home/lgx/data/whitebox_attack_cmupose/train_labeled_npz"
        output_labeled_img_path = "/home/lgx/data/whitebox_attack_cmupose/train_labeled_images"
        if not os.path.exists(output_npz_path):
            os.makedirs(output_npz_path)
        if not os.path.exists(output_labeled_img_path):
            os.makedirs(output_labeled_img_path)
        

        preprocess(input_images_path, yolox_path,output_npz_path, output_labeled_img_path,scale,pad_x, pad_y, dataset_resolution_format='gen1')
        pass

if __name__ == "__main__":
    main()
