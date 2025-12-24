'''
read and handle data set
'''

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from Debug.functional import showMessage
from smpl_loading import *
import random
import math
import imageio
import cv2

class DatasetFunction(Dataset):
    def __init__(self):
        #super().__init__()
        self.extensions = ['.jpg', '.png', '.JPG', '.PNG']
    
    def loadImage(self, fileName):
        return Image.open(fileName)
        
    def loadNpy(self, fileName):
        return np.load(fileName)
    
    def getImagePath(self, root, baseName, extension=None):
        if(extension is not None):
            return os.path.join(root, '{}{}'.format(baseName, extension))
        else:
            return os.path.join(root, '{}'.format(baseName))
    
    def getImageBaseName(self, fileName):
        return os.path.basename(os.path.splitext(fileName)[0])

class MyDataSet(DatasetFunction):
    def __init__(self, args, transform, debug_mode=False, train_mode=False):
        self.args = args
        
        self.reshape_size = args.size
        
        self.transform = transform
        self.debug_mode = debug_mode
        self.train_mode = train_mode
        self.frame_num = args.seq_len * (args.bins +1)
        print("self.frame_num",self.frame_num)
        self.simu_data_path = args.data_simu
        assert os.path.exists(self.simu_data_path), "{} not exists !".format(self.simu_data_path)
        
        video_list = os.listdir(self.simu_data_path)
        interval_num =1 #self.frame_num
        self.file_list = []
        for video in video_list:
            if video[-1] == '\n':
                video = video[:-1]
            index = 0
            interval = self.frame_num-2
            frames = (os.listdir(os.path.join(self.simu_data_path , video)))
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [f"{img_index:05d}"+'.png' for img_index in frames]
            while index + (interval + 1) * interval_num + 2 < len(frames) - 0:
                videoInputs = [frames[i] for i in range(index, index + (1 + interval) * interval_num + 1,1)]
                videoInputs = [os.path.join(self.simu_data_path,video, f) for f in videoInputs]
  
                self.file_list.append(videoInputs)

                index += interval+2
       # print("self.file_list",self.file_list)

    def __getitem__(self, index):
        #index = 1
       # print("index",index)
       # print("self.file_list.len",len(self.file_list))
        imageName_list = self.file_list[index]
        #irradiance_map = self.loadNpy(imageName)[0]
        #image = self.loadImage(imageName).convert('RGB')
        irradiance_maps = [np.array(self.loadImage(imageName).convert('L')) for imageName in imageName_list]#gray
        irradiance_maps = np.stack(irradiance_maps,axis=0)
        #print("irradiance_maps.sahpe",irradiance_maps.shape)
        if self.train_mode:
            irradiance_maps = self.transform(irradiance_maps)
            return irradiance_maps #, real_event_image, real_event_map,real_event_image2
        else:
            image, irradiance_map = self.transform(irradiance_maps)
            return image, irradiance_map

    def __len__(self):
        return len(self.file_list)

class PoseDataSet(DatasetFunction):
    def __init__(self, args, transform, debug_mode=False, train_mode=False,phase = None):
        self.args = args
        self.reshape_size = args['size']
        self.transform = transform
        self.debug_mode = debug_mode
        self.train_mode = train_mode
        self.frame_num = self.args['seq_len'] * (self.args['bins']) + 1
        self.smpl_model_path = args['smpl_model_dir']
        print("self.frame_num", self.frame_num)
        self.simu_pose_path = args['data_simu_pose']
        self.labeled_simu_pose = args['labeled_simu_pose']
        self.original_uv_img_path = args['original_uv_img']
        #self.
        self.mask_path = self.args['mask_path']#
        self.vis_parts = self.args['mask_part']
        if phase=='train':
            human_size_list =["middle"] # , "middle"] #,"small"]
        else:
            human_size_list =['large',"middle","small"]

        self.T_dict = {"large":1.5, "middle":1.75, "small":2}
        self.cell_size = self.args['cell_size']
        combined_mask = None
        for vis_part in self.vis_parts:
            mask_path = os.path.join(self.mask_path,"mask_"+vis_part+".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if combined_mask is None:
                combined_mask = np.zeros_like(mask)
            
            combined_mask = cv2.bitwise_or(combined_mask, mask)


        print(combined_mask.shape,combined_mask.max(),combined_mask.min())# = cv2.bitwise_or(combined_mask, mask)

        h,w = combined_mask.shape[0],combined_mask.shape[1]
        self.mask = torch.from_numpy(combined_mask.astype(np.float32)/255.).reshape(h,w,1)#.float().reshape()

        assert os.path.exists(self.simu_pose_path), "{} not exists !".format(self.simu_pose_path)
        poses_list = os.listdir(self.simu_pose_path)
        self.file_list = []
        self.labeled_list = []
        self.t_val_list = []
        # if phase == "train":
        #     step=1
        # else:
        step = self.frame_num
        timelens_all = 0
        for pose_name in poses_list:
            video_name  = pose_name[:2]
            complete_pose_path = os.path.join(self.simu_pose_path, pose_name)
            for human_size in human_size_list:
                complete_label_path = os.path.join(self.labeled_simu_pose,f"labels_{human_size}",video_name,pose_name)
                verts_sequence, faces =load_pose_from_file(complete_pose_path)

                timelens = len(verts_sequence)
                timelens_all += timelens
                for i in range(0,timelens,step):
                    start_index = i
                    end_index = start_index + self.frame_num
                    if end_index <= timelens-1:
                        self.file_list.append((complete_pose_path,start_index,end_index))# 31 frames
                        self.labeled_list.append((complete_label_path,start_index,end_index)) # 31 frames
                        self.t_val_list.append(self.T_dict[human_size])
                    else:
                        break
        print("timelens_all",timelens_all)
        print("len(self.file_list)",len(self.file_list))
        print("len(self.labeled_list)",len(self.labeled_list))
        

        self.uv_obj_path = args['uv_obj_path']
        verts, faces, aux = load_obj(
            self.uv_obj_path,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
        self.verts_uvs = aux.verts_uvs #.to(device)
        self.faces_uvs = faces.textures_idx #to(device)

    def __getitem__(self, index):
        pose_seq_path = self.file_list[index][0]
        start_index = self.file_list[index][1]
        end_index = self.file_list[index][2]
        
        labeled_seq_path = self.labeled_list[index][0]
        labeled_list =  np.load(labeled_seq_path)['labels']
        t_val = self.t_val_list[index]
        #print("t_val ",t_val,"path ",labeled_seq_path)
        verts_sequence, faces =load_pose_from_file(pose_seq_path)
             
        verts_seq = verts_sequence[start_index:end_index]
        step = self.args['bins']
        labels = labeled_list[start_index:end_index-1:step]
        #print("pose_seq",pose_seq_path,"start_index",start_index,"end_index",end_index)
        assert len(verts_seq) == self.frame_num, "error"
        faces = torch.from_numpy(np.ascontiguousarray(faces)).float()
        verts_seq = np.stack(verts_seq,axis=0)
        verts_seq = torch.from_numpy(np.ascontiguousarray(verts_seq)).float()
        #uv_input = cv2.imread(self.original_uv_img_path).astype(np.float32)/255.
       # uv_input = np.ones_like(uv_input)#/1
       # uv_input = uv_input[..., [2, 1, 0]] #BGR -> RGB
        
        # gray_input = True
        # if gray_input:
        #     r, g, b = uv_input[..., 0],uv_input[..., 1], uv_input[..., 2]
        #     uv_input_gray_image = 0.299 * r + 0.587 * g + 0.114 * b
        #     uv_input = np.expand_dims(uv_input_gray_image, axis=-1)
        count = (int)(math.ceil(1024/self.cell_size))

        uv_input = np.ones((1,count*count),dtype=np.float32)
        #uv_input = np.random.rand(1,count*count).astype(np.float32)
        return {'verts_seq': verts_seq, 'faces': faces, 'verts_uvs':self.verts_uvs , 'faces_uvs':self.faces_uvs,'uv_input': uv_input , 'labels':labels, 'vis_mask':self.mask, "t_val": t_val}

    def __len__(self):
        return len(self.file_list)