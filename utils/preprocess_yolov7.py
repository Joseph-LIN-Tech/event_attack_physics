import glob
from call_yolo import *
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
from pytorch3d.io import load_obj, save_obj

import debugpy
# debugpy.listen(5679)
# debugpy.wait_for_client()

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

    verts = []
    for i in range(time_length):
        vertices=c2c(body_pose_beta.v[i])# vertices for each timestamp
        #print(f"{i},vertices=",vertices)
        verts.append(vertices)
    return verts, faces 



# 渲染SMPL模型
def render_smpl_model(verts, faces, verts_uvs, faces_uvs, texture_image,H,W,device,T):

    print("T",T)
    textures = TexturesUV(maps=texture_image, faces_uvs=[faces_uvs], verts_uvs=[verts_uvs])
    smpl_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    R =  torch.tensor([[0,0,-1],[1,0,0],[0, 1, 0]],device=device)
    T = torch.tensor([0,0,T],device=device)
    cameras = PerspectiveCameras(device=device, R=R[None], T=T[None])

    raster_settings = RasterizationSettings(image_size=(H,W), blur_radius=0.0, bin_size = 0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    images = renderer(smpl_mesh).to(device)
    return images[0, ..., :3]


def preprocess(input_images_path, yolo_path,output_npz_path_, output_labeled_img_path_, dataset_resolution_format='gen1'):
    '''
    Purpose: to create the label of human for the detection
    '''
    video_list = os.listdir(input_images_path)
    for video in video_list:
        video_name = str(video)

        video_path = os.path.join(input_images_path,video)
        output_npz_path = os.path.join(output_npz_path_,video_name)
        if not os.path.exists(output_npz_path):
            os.makedirs(output_npz_path)

        output_labeled_img_path = os.path.join(output_labeled_img_path_,video_name)
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
            last_array = None
            for img_path in rgb_img_path_sorted:
                t = (int)(index*time_interval_ms*1000)  #   us 
                return_val =callback_yolov7(yolo_path, img_path,output_labeled_img_file,cuda_id="2")

                #box_pattern = [r"\[\[.*?\]\]",r"\[\[.*?\]\n"]
                pattern = r'result:\s*\(([^)]+)\)'
                box_coord = find_pattern(return_val,pattern)
                print("box_Coord",box_coord)
                if box_coord is None:
                    array_list.append(last_array)
                else:
                    if dataset_resolution_format == "gen1":
                        #class_id = box_coord[0]
                        x_center = box_coord[1]
                        y_center = box_coord[2]
                        w = box_coord[3]
                        h = box_coord[4]
                        confidence_score = 0 #box_coord[5]  minimize the confidence_score

                        x_lefttop = x_center - w/2 
                        y_lefttop = y_center - h/2   

                    assert (w>0 and h>0), print("h and w must larger than 0")
                    class_id = 1  # car is 0, human is 1
                    #track_id = id
       

                    array = [(int)(t),(int)(x_lefttop),(int)(y_lefttop),(int)(w),(int)(h),(int)(class_id),confidence_score,track_id]
                    print("array",array)
                    last_array = array
                    array_list.append(array)
                index+=1


            # 保存NumPy数组为.npy文件
            array_path = os.path.join(output_npz_path,f'{pose_name}.npz')
            print("array_path",array_path)
            np.savez(array_path, labels=array_list)

            sequence_labels = np.load(str(array_path))
            print("sequence_labels",sequence_labels['labels'])
            #break


def preprocess_video(input_video_path, yolo_path,output_npz_path_, output_labeled_video_path, dataset_resolution_format='gen1',cuda_id="0"):
    '''
    Purpose: to create the label of human for the detection
    '''
    video_list = os.listdir(input_video_path)
    for video in video_list:
        video_name = str(video)
        video_path = os.path.join(input_video_path,video)
        # output_npz_path = os.path.join(output_npz_path_,video_name)
        # if not os.path.exists(output_npz_path):
        #     os.makedirs(output_npz_path)
        pose_name = video_name.replace(".mp4","")

        if not os.path.exists(output_labeled_video_path):
            os.makedirs(output_labeled_video_path)
        output_pose_path = os.path.join(output_labeled_video_path,pose_name)
        output_pose_labels_path = os.path.join(output_pose_path,"labels")
        if not os.path.exists(output_pose_path):
            os.makedirs(output_pose_path)
        if not os.path.exists(output_pose_labels_path):
            os.makedirs(output_pose_labels_path)

        callback_yolov7(yolo_path, video_path, output_pose_path, cuda_id)

def generate_labels_4_rvt(input_path,output_npz_path):
    pose_list = os.listdir(input_path)
    pose_list = sorted(pose_list)
    track_id = 0
    for pose in pose_list:
        track_id += 1 
        pose_name = pose
        video_name = str(pose_name[:2])
        output_labels = os.path.join(output_npz_path,video_name)
        if not os.path.exists(output_labels):
            os.makedirs(output_labels)
        
        pose_path = os.path.join(input_path,pose_name)
        label_path = os.path.join(pose_path,"labels")

        # Get all .txt files from the directory
        txt_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
        # Function to extract the number from the filename
        def extract_number(filename):
            match = re.search(r'poses_(\d+)', filename)
            return int(match.group(1)) if match else -1
        # Sort the files based on the extracted number
        txt_files.sort(key=extract_number)
        array_list = []
        index = 0
        fps = 120
        time_interval_ms = 1000/fps # ms

        for txt_file in txt_files:
            t = (int)(index*time_interval_ms*1000)  #   us 
            file_path = os.path.join(label_path, txt_file)
            with open(file_path, 'r') as file:
                content = file.readline().strip()
                # Convert the content into a list array
                data = content.split()
                # Convert string elements to appropriate types (int or float)
                box_coord = [float(item) if '.' in item else int(item) for item in data]
                #if dataset_resolution_format == "gen1":
                #class_id = box_coord[0]
                x_center = box_coord[1]
                y_center = box_coord[2]
                w = box_coord[3]
                h = box_coord[4]
                confidence_score = box_coord[5]  
                x_lefttop = x_center - w/2 
                y_lefttop = y_center - h/2   

                assert (w>0 and h>0), print("h and w must larger than 0")
                class_id = 1  # car is 0, human is 1
                #track_id = id
    

                array = [(int)(t),(int)(x_lefttop),(int)(y_lefttop),(int)(w),(int)(h),(int)(class_id),confidence_score,track_id]
                print("array",array)
                # last_array = array
                array_list.append(array)
                index += 1
        # 保存NumPy数组为.npy文件
        array_path = os.path.join(output_labels,f'{pose_name}.npz')
        print("array_path",array_path)
        np.savez(array_path, labels=array_list)

        sequence_labels = np.load(str(array_path))
        print("sequence_labels",sequence_labels['labels'])



def render_images(verts, faces, verts_uvs, faces_uvs, uv_texture, output_path, H,W, device,T):
    faces = torch.tensor(faces, dtype=torch.long, device=device)#.unsqueeze(0)

    for i in range(len(verts)):
        verts_sub = torch.tensor(verts[i],  device=device).squeeze(0)    
        rendered_image = render_smpl_model(verts_sub, faces, verts_uvs, faces_uvs, uv_texture,H,W,device,T)*255.0
        image_array = rendered_image.detach().cpu().numpy().astype(np.uint8)  # 将浮点数转换为8位无符号整数
        print("before, image_array.shape",image_array.shape)
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(image_array)
      
        output_img_name = os.path.join(output_path,f"{i:05d}.png")
        print("save img: ",output_img_name)
        image.save(output_img_name)

def load_pose_and_rendering(smpl_params_path,verts_uvs,faces_uvs,output_path,H,W, device,T):
    video_list = os.listdir(smpl_params_path)
    uv_texture = torch.ones((1,512,512,3),device=device)
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
            png_list = os.listdir(output_pose_image_path)
            if len(png_list) == len(verts):
                print(pose_file, "continue")
                continue
            render_images(verts, faces, verts_uvs, faces_uvs,uv_texture, output_pose_image_path,H,W,device,T)
   

def convert_Img2Mp4(source_path,output_path):
    video_list = os.listdir(source_path)
    for video in video_list:
        video_path = os.path.join(source_path,video)
        pose_files = os.listdir(video_path)
        for pose in pose_files:
            image_folder = os.path.join(video_path,pose)
            output_video = os.path.join(output_path,pose+".mp4")
            images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
            images.sort()  # Ensure the images are sorted in the correct order

            # Read the first image to get the dimensions
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
            framerate = 120
            video = cv2.VideoWriter(output_video, fourcc, framerate, (width, height))

            # Iterate through images and write them to the video
            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))

            # Release the VideoWriter object
            video.release()
            cv2.destroyAllWindows()

            print(f"Video saved as {output_video}")


def main():
    render_images = False
    convert_2_mp4 = False
    labeling_images = False
    labeling_video = False
    labeling_4_rvt = True
    human_size = "small"
    T_dict = {"large":1.25, "middle":1.5, "small":1.75}
    
    T = T_dict[human_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    current_path = os.getcwd()
    print("当前工作路径:", current_path)

    yolo_path = f"{current_path}/checkpoints/yolov7"
    data_path = "/home/lgx/data/whitebox_attack_cmupose"
    smpl_params_path = f'{data_path}/whitebox_attack'  # 你的SMPL模型文件路径
    #uv_texture_path = '/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/smpl_uv.png'  # 你的UV贴图文件路径
    #uv_texture_path = "/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/patch1.png"
    uv_obj_path =  f"{current_path}/Data/smpl_uv.obj"
    #smpl_model_path = r"/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data"
    name_T = str(T).replace(".","_")
    output_rendered_images = f"{data_path}/output_T_{name_T}"
    H, W= 240,304
    

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
        load_pose_and_rendering(smpl_params_path,verts_uvs,faces_uvs,output_rendered_images, H,W, device,T)
    
    if convert_2_mp4:
        source_path = f"{data_path}/images_{human_size}"
        output_path = f"{data_path}/video_{human_size}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        convert_Img2Mp4(source_path,output_path)

    if labeling_images:
        input_images_path = f"{data_path}/output_T_{name_T}"
        output_npz_path = f"{data_path}/output_npz_T_{name_T}"
        output_labeled_img_path = f"{data_path}/output_label_img_T_{name_T}"
        if not os.path.exists(output_npz_path):
            os.makedirs(output_npz_path)
        if not os.path.exists(output_labeled_img_path):
            os.makedirs(output_labeled_img_path)
        
        preprocess(input_images_path, yolo_path,output_npz_path, output_labeled_img_path,dataset_resolution_format='gen1')

    if labeling_video:
        human_size_list = ["large","middle","small"]
        for human_size in human_size_list:
            input_video_path = f"{data_path}/video_{human_size}"
            output_npz_path = f"{data_path}/labels_{human_size}"
            output_labeled_video_path = f"{data_path}/video_rawlabels_{human_size}"
            if not os.path.exists(output_npz_path):
                os.makedirs(output_npz_path)
            if not os.path.exists(output_labeled_video_path):
                os.makedirs(output_labeled_video_path)

            preprocess_video(input_video_path, yolo_path,output_npz_path, output_labeled_video_path,dataset_resolution_format='gen1')
    
    if labeling_4_rvt:
        #convert the labels to the format compatible with rvt
        human_size_list = ['small',"large","middle"]
        for human_size in human_size_list:
            input_path = f"{data_path}/video_rawlabels_{human_size}"
            output_npz_path = f"{data_path}/labels_{human_size}"

            generate_labels_4_rvt(input_path,output_npz_path)

        pass

if __name__ == "__main__":
    main()
