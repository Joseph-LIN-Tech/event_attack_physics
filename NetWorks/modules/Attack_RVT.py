import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import imageio
import random
import time
import math
from torch.utils import model_zoo
from torchvision import models


from NetWorks.utils import *
from NetWorks.__init__ import loadPretrainedParameter
from torch.nn.functional import pad
import pytorch_lightning as pl
from detection import Module as rnn_det_module
import torchvision.transforms.functional as TF

from config_rvt.modifier import dynamically_modify_train_config
from data_rvt.utils.types import DatasetSamplingMode
from loggers_rvt.utils import get_wandb_logger, get_ckpt_path
from modules_rvt.utils.fetch import fetch_data_module, fetch_model_module

from Criterion.criterion import Criterion
from scipy.optimize import linear_sum_assignment as linear_assignment
from tensorboardX import SummaryWriter
from smpl_loading import *
from torch import Tensor
#from utils.km import KuhnMunkres,Bayes
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import math

def freeze_bn_layers(model):
    model.eval()

def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    model_str = config.model.name
    if model_str == 'rnndet':
        return rnn_det_module(config)
    raise NotImplementedError

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,alpha):
        hard_binarized = (input > 0.5).float()
        soft_binarized = alpha * hard_binarized + (1 - alpha) * input
        return soft_binarized
        #return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output,None

class STEBinarizer(nn.Module):
    def __init__(self):
        super(STEBinarizer, self).__init__()

    def forward(self, x,alpha):
        return STEFunction.apply(x,alpha)

class UV_Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cell_size,binarize):
        super(UV_Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5) 
        self.binarize = binarize
        if binarize:
            self.binarizer = STEBinarizer()
        self.cell_size = cell_size
        self.upsample = nn.Upsample(scale_factor=self.cell_size, mode='nearest')

    def forward(self, x,alpha=1.0):
        b, c, n = x.shape
        x= x.reshape(b,c*n)
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        if self.binarize:
            x = self.binarizer(x,alpha)
        x = x.view(1, 1, int(math.sqrt(n)), int(math.sqrt(n))) 
        x = self.upsample(x)
        return x
    

import math
class attack_rvt(nn.Module):
    def __init__(self, args,config):
        super(attack_rvt,self).__init__()
        self.args = args['network']
        self.patch = self.args['patch']
        self.patch_size = self.args['size'][0]//self.args['patch']
        self.uv_h = self.args['uv_size'][0]
        self.uv_w = self.args['uv_size'][1]
        self.cell_size = self.args['cell_size']
        print("self.cell_size",self.cell_size)
        self.if_binary = self.args['if_binary']
        print("self.if_binary",self.if_binary)
        ckpt_path = Path(config.checkpoint)
        dynamically_modify_train_config(config)
        # ---------------------
        # Model
        # ---------------------
        module = fetch_model_module(config=config)
        self.rvt_model = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        rvt_model_stage = "test"
        self.rvt_model.setup(rvt_model_stage)
        for param in self.rvt_model.parameters():#frozen
            param.requires_grad = False
        
        self.rvt_model.eval()
        freeze_bn_layers(self.rvt_model)

        count = (int)(math.ceil(1024/self.cell_size))
        print("count",count)
        input_dim = count * count   
        hidden_dim = count*count*2   
        output_dim = count*count

        self.uv_net = UV_Generator(input_dim, hidden_dim, output_dim,self.cell_size,self.if_binary)

        self.white_image = torch.ones((1,1,1024,1024))
     
        self.H,self.W = self.args['reshape_size'][0],self.args['reshape_size'][1]
        
        self.output_rgb_path = self.args['output_rgb_path']
        if not os.path.exists(self.output_rgb_path):
            os.makedirs(self.output_rgb_path)

        self.debug_mode = self.args['debug']
        self.resizeShape = self.args['size']
        
        self.bins = self.args['bins']
        self.seq_len = self.args['seq_len']
        
        self.pre_pos_threshold = self.args['contrast_threshold_pos']
        self.pre_neg_threshold = self.args['contrast_threshold_neg']
    
        self.threshold_C_pos = self.pre_pos_threshold
        self.threshold_C_neg = self.pre_neg_threshold
        self.ev_scale = 10/(math.floor(math.log(255)/self.threshold_C_neg))
        self.cut_off = 10
        self.ev = self.cut_off/27
    
    def save_img(self,input_tensor,output_rgb_path):
        '''
        range [0,255]
        '''
        image_array = torch.clamp(input_tensor,0,255)
        image_array = image_array.detach().cpu().numpy().astype(np.uint8)  
        image = Image.fromarray(image_array)
        image.save(output_rgb_path)

    def ln_map(self, map):
        '''
        input : map [0,255]
        '''
        new_map = map.clone().to(map.device)
        new_map[map < self.args['log_threshold']] = map[map < self.args['log_threshold']]/self.args['log_threshold']*math.log(self.args['log_threshold'])
        new_map[map >= self.args['log_threshold']] = torch.log(map[map >= self.args['log_threshold']])

        return new_map
    

    def rgb_to_y(self,image):
        if not isinstance(image, Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-1] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        r  = image[..., 0]
        g  = image[..., 1]
        b  = image[..., 2]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        # u = -0.147 * r - 0.289 * g + 0.436 * b
        # v = 0.615 * r - 0.515 * g - 0.100 * b
        return y
    
    def video2event(self, images):
        # images in [0,255]        
        threshold_C_pos = self.threshold_C_pos
        threshold_C_neg = self.threshold_C_neg
        ev_seq_torch_list =[]
        b,c,h,w = images.size()
        bins = self.bins
        assert images.shape[1] == (self.seq_len * (bins)+1), "error"
        for j in range(self.seq_len):
            images_seq = images[:,j*bins:(j+1)*bins+1] # [0,255]
            #print("j*(bins+1)",j*bins,"(j+1)*(bins+1)",(j+1)*bins+1)
            images_origin = images_seq[:,0:1]
            irradiance_maps_0 = self.ln_map(images_origin)
            #random_sample = (torch.rand([1,1,h,w])*0.2).to(images.device)
            #irradiance_maps_0 -= random_sample

            irradiance_maps =torch.zeros((irradiance_maps_0.size()[0],bins+1,irradiance_maps_0.size()[2],irradiance_maps_0.size()[3])).to(images.device)
            for i in range(1,bins+1):
                #print("i",i)
                images_next = images_seq[:,i:i+1]#[:,:,min_x:max_x,min_y:max_y]
                lpLogFrame1 = self.ln_map(images_next)
                irradiance_maps[:,i:i+1] = lpLogFrame1-irradiance_maps_0 # diff between i-th and 0th, irradiance_maps indicate the relative values
                
            threshold_C_pos_cp = threshold_C_pos
            irradiance_maps_pos_trunc = torch.div(F.relu(irradiance_maps), threshold_C_pos_cp)*threshold_C_pos_cp
           # irradiance_maps_pos_trunc = torch.floor(F.relu(irradiance_maps)/threshold_C_pos_cp)*threshold_C_pos_cp
            irradiance_values_pos = irradiance_maps_pos_trunc/ threshold_C_pos  # irradiance_values_pos: transfer from values to multipler, located in [0, +m]
            
            threshold_C_neg_cp = threshold_C_neg
            irradiance_maps_neg_trunc = (torch.div(F.relu(-irradiance_maps), threshold_C_neg_cp)) *threshold_C_neg_cp
            #irradiance_maps_neg_trunc = torch.floor(F.relu(-irradiance_maps)/threshold_C_neg_cp) *threshold_C_neg_cp

            irradiance_values_neg = irradiance_maps_neg_trunc / threshold_C_neg # irradiance_values_neg: transfer from values to multipler which located in [0, +n]

            irradiance_values = irradiance_values_pos - irradiance_values_neg # => irradiance_values: pos_pixel: >0   neg_pixel: <0
            diff_img = irradiance_values[:,1:bins+1] - irradiance_values[:,0:bins]# diff between consecutive frame: value >0 => pos, value <0 => neg
            diff_img=diff_img-((diff_img>0)*(irradiance_maps[:,1:bins+1]<=0)).float()+((diff_img<0)*(irradiance_maps[:,1:bins+1]>=0)).float()# add residual
     
            output_pos = (F.relu(diff_img)) 
            output_neg = (F.relu(-diff_img)) 
            ev_seq_torch = torch.cat((output_neg,output_pos), 1) 
            ev_seq_torch = (ev_seq_torch*self.ev_scale)
            ev_seq_torch = ev_seq_torch.to(images.device)
            ev_seq_torch_list.append(ev_seq_torch)

        return ev_seq_torch_list

    def get_labels_as_tensor(self, input_labels,format_,device):
        '''
        input labels format orders:  [(int)(t),(int)(x_lefttop),(int)(y_lefttop),(int)(w),(int)(h),(int)(class_id),confidence_score,track_id]
        '''
        labels_len = len(input_labels) 
        if format_ == 'yolox':
            out = torch.zeros((labels_len,5), dtype=torch.float32, device=device)
            if labels_len == 0:
                return out
            out[:, 0] = input_labels[:,5] #class_id
            out[:, 1] = input_labels[:,1] + 0.5 * input_labels[:,3] # x_center
            out[:, 2] = input_labels[:,2] + 0.5 * input_labels[:,4] #y_center
            out[:, 3] = input_labels[:,3] # w
            out[:, 4] = input_labels[:,4] # h
            return out
        else:
            raise NotImplementedError
    

    def horizontal_flip(self,image_batch):
        return TF.hflip(image_batch)
        
    def get_labels_as_batched_tensor(self,input_labels,format_,device):
        input_labels = input_labels[0] 
        num_object_frames = len(input_labels)
        assert num_object_frames > 0
        max_num_labels_per_object_frame = 1
        assert max_num_labels_per_object_frame > 0
        if format_ == 'yolox':
            tensor_labels = []
            for labels in input_labels:
                labels = labels.reshape(1,len(labels))
                obj_labels_tensor = self.get_labels_as_tensor(labels,format_,device)
                num_to_pad = max_num_labels_per_object_frame - len(labels)
                padded_labels = pad(obj_labels_tensor, (0, 0, 0, num_to_pad), mode='constant', value=0)
                tensor_labels.append(padded_labels)
            tensor_labels = torch.stack(tensors=tensor_labels, dim=0)

            return tensor_labels
        else:
            raise NotImplementedError    

    def forward(self,verts_seq,faces,verts_uvs,faces_uvs,uv_input,labels,metrics=None,vis_mask=None,T_val=2,alpha=1):
        freeze_bn_layers(self.rvt_model)
        self.vis_mask = vis_mask.permute(0,3,1,2) 
        B,C,H,W = 1,1,1024,1024
        device = verts_seq.device
        
        #labels_yolox = self.get_labels_as_batched_tensor(labels,"yolox",device)
        self.white_image = self.white_image.to(device)

        self.uv_texture = self.uv_net(uv_input,alpha)

        self.uv_texture = self.uv_texture[:,:,:H,:W]
        self.uv_texture_temp = self.uv_texture.permute(0, 2, 3, 1)
        self.uv_texture_temp = self.uv_texture_temp.repeat(1, 1, 1, 3)

        self.save_img(self.uv_texture_temp[0]*255,os.path.join(self.output_rgb_path.replace("RGB",'imm'),f"before_mask_uv_texture.png"))
        self.uv_texture = self.vis_mask *self.uv_texture + (1- self.vis_mask)*self.white_image[:,:,:H,:W]
        
        self.vis_mask_temp = self.vis_mask.permute(0, 2, 3, 1)
        self.vis_mask_temp = self.vis_mask_temp.repeat(1, 1, 1, 3)

        self.save_img(self.vis_mask_temp[0]*255,os.path.join(self.output_rgb_path.replace("RGB",'imm'),f"vis_mask.png"))

        self.uv_texture = self.uv_texture.permute(0, 2, 3, 1)
        self.uv_texture  = self.uv_texture.repeat(1, 1, 1, 3)

        self.save_img(self.uv_texture[0]*255,os.path.join(self.output_rgb_path.replace("RGB",'imm'),f"uv_texture.png"))
        
        B,T,N,C = verts_seq.size()
        R1=-1
        R2=1
        R3=1
        render_image_arr = torch.zeros((B,T,self.H,self.W,3)).to(device)
        if self.training:
            random_Tval = True and random.random() < 0.4
            if random_Tval:
                delta = 0.25
                T_val = T_val + random.uniform(-delta, delta)
           
            random_rorate = True and random.random() < 0.4
            if random_rorate:
                delta = 0.1
                R1 += random.uniform(-delta, delta)
                R2 += random.uniform(-delta, delta)
                R3 += random.uniform(-delta, delta)
                #print(R1,R2,R3)
        
        for b in range(B):
            for i in range(T):
                # render the 2d images from the given learnable uv_texture, which is represented by self.uv_texture
                rendered_img = render_smpl_model(verts_seq[b][i].to(device), faces[b].to(device), verts_uvs[b].to(device),\
                                                    faces_uvs[b].to(device), self.uv_texture, self.H, self.W, device,R1,R2,R3,T_val)*255.0
                render_image_arr[b:b+1,i:i+1,:,:,:] = rendered_img
                
                self.save_img(rendered_img,os.path.join(self.output_rgb_path,f"rgb_b{b}_i{i}.png"))
        
        # if self.training:
            

        irradiance_maps = self.rgb_to_y(render_image_arr) 

        #get the event voxel from the given images with y channels
        simulated_event_maps  = self.video2event(irradiance_maps)
        if self.training:
            predict, _ = self.rvt_model.val_test_step(ev_tensor_sequence=simulated_event_maps,labels_yolox=None,obj_labels= labels,head_loss=True)
            predict  = predict.view(-1, predict.size(-1))
            target_class = 1 # car :0, human 1
            object_confidence = predict[:, 4].mean()
            class_confidence = predict[:,5+target_class].mean()
            return object_confidence, class_confidence
        else:
            rtval  = self.rvt_model.val_test_step(ev_tensor_sequence=simulated_event_maps,labels_yolox=None,obj_labels= labels,metrics=metrics,head_loss=False)

            return rtval, self.uv_texture
        
    def inference(self,verts_seq,faces,verts_uvs,faces_uvs,uv_texture,labels,metrics=None,vis_mask=None,T_val=2):
        #print("T_val",T_val)
        freeze_bn_layers(self.rvt_model)
        self.vis_mask = vis_mask.permute(0,3,1,2) 
        B,C,H,W = 1,1,1024,1024
        device = verts_seq.device
        
        #labels_yolox = self.get_labels_as_batched_tensor(labels,"yolox",device)
        self.white_image = self.white_image.to(device)
    
        self.uv_texture = uv_texture.to(device)

        self.save_img(self.uv_texture[0]*255,os.path.join(self.output_rgb_path.replace("RGB",'imm'),f"test_uv_texture.png"))
        B,T,N,C = verts_seq.size()
        
        render_image_arr = torch.zeros((B,T,self.H,self.W,3)).to(device)
        R1=-1
        R2=1
        R3=1
        for b in range(B):
            for i in range(T):
                # render the 2d images from the given learnable uv_texture, which is represented by self.uv_texture
                rendered_img = render_smpl_model(verts_seq[b][i].to(device), faces[b].to(device), verts_uvs[b].to(device),\
                                                    faces_uvs[b].to(device), self.uv_texture, self.H, self.W, device,R1,R2,R3,T_val)*255.0
                render_image_arr[b:b+1,i:i+1,:,:,:] = rendered_img
                self.save_img(rendered_img,os.path.join(self.output_rgb_path,f"rgb_b{b}_i{i}.png"))

        irradiance_maps = self.rgb_to_y(render_image_arr) 

        #get the event voxel from the given images with y channels
        simulated_event_maps  = self.video2event(irradiance_maps)
        if self.training:
            predict, _ = self.rvt_model.val_test_step(ev_tensor_sequence=simulated_event_maps,labels_yolox=None,obj_labels= labels)
            predict  = predict.view(-1, predict.size(-1))
            target_class = 1 # car :0, human 1
            object_confidence = predict[:, 4].mean()
            class_confidence = predict[:,5+target_class].mean()
            return object_confidence, class_confidence
        else:
            attack_success  = self.rvt_model.val_test_step(ev_tensor_sequence=simulated_event_maps,labels_yolox=None,obj_labels= labels,metrics=metrics)

            return attack_success
        

    def cal_AP_after_epoch(self):
        freeze_bn_layers(self.rvt_model)
        rvt = self.rvt_model.on_test_epoch_end()
        return rvt, self.uv_texture