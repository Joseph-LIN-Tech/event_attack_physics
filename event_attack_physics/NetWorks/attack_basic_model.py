import logging
from collections import OrderedDict
import cv2
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import NetWorks.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from NetWorks.modules.loss import CharbonnierLoss, LapLoss
from pdb import set_trace as bp
import random
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('base')
import os.path as osp
import NetWorks.modules.Attack_RVT as Attack_RVT
from enum import Enum, auto

class Mode(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()



class attack_basic_model(BaseModel):
    def __init__(self, args,config):
        super(attack_basic_model, self).__init__(args,config)
        
        self.lrs = []
        self.obj_confs = []
        self.cls_confs = []

        # define network and load pretrained models
        self.netG = Attack_RVT.attack_rvt(args,config).to(self.device)     
        self.opt = args
        self.load()

        if self.is_train:
            self.netG.train()
            #### loss
            self.lambda_w =self.opt['train']['lambda']

            #### optimizers
            wd_G =  1e-5
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
                    

            self.optimizer_G = torch.optim.Adam(optim_params, lr=self.opt['train']['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(self.opt['train']['beta1'], self.opt['train']['beta2']))
            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, T_period=self.opt['train']['T_period'], eta_min=self.opt['train']['eta_min'],
                        restarts=self.opt['train']['restarts'], weights=self.opt['train']['restart_weights']))
            self.log_dict = OrderedDict()

    def dist_validation_uvmap(self, dataloader, uv_texture,current_iter, tb_logger, save_img,metrics):
        avg_attack_success_rate = 0
        seq_attack_success_rate =0
        
        # for avg_attack_success_rate
        total_predicted_list = []
        human_detected_list = []

        #for seq_attack_success_rate
        seq_human_deteced_all = []
        if "AP" in metrics:
            assert len(metrics)==1,"AP must be tested alone"
            with tqdm(total=len(dataloader), desc=f'Validate',leave=True) as pbar:
                for index, val_data in enumerate(dataloader):
                    self.feed_data(val_data)
                    rtval =self.test_uvmap(uv_texture,output=True,metrics="AP")
                    pbar.update(1)  # 
            
            metrics , uv_texture = self.netG.cal_AP_after_epoch()
            AP = metrics['AP']
            AP_50 = metrics['AP_50']
            AP_75 = metrics['AP_75']
            AP_S = metrics['AP_S']
            AP_M = metrics['AP_M']
            AP_L = metrics['AP_L']
            logger.info('AP : {}, AP_50 : {}, AP_75 : {}, AP_S : {}, AP_M : {}, AP_L: {}'.format(AP,AP_50,AP_75,AP_S,AP_M,AP_L))
            return metrics,uv_texture
            
        else:
            with torch.no_grad():
                with tqdm(total=len(dataloader), desc=f'Validate',leave=True) as pbar:
                    for index, val_data in enumerate(dataloader):
                        self.feed_data(val_data)
                        for metric in metrics :
                            if metric == "avg_attack_success_rate":
                                rtval =self.test_uvmap(uv_texture,output=True,metrics=metric)
                                human_detected, total_detected = rtval[0],rtval[1] 
                                #print(f"avg:  human_detected={human_detected}, total_detected={total_detected}")
                                total_predicted_list.append(total_detected)
                                human_detected_list.append(human_detected)
                                
                            if metric == "seq_attack_success_rate":
                                rtval =self.test_uvmap(uv_texture, output=True,metrics=metric)
                                seq_human_detected,total_detected = rtval[0], rtval[1]
                                #print(f"seq:  human_detected={seq_human_detected}, total_detected={total_detected}")
                                seq_human_deteced_all.append(seq_human_detected)

                        

                        pbar.update(1)  # 
                        
                        avg_attack_success_rate = 1-(sum(human_detected_list) / sum(total_predicted_list)) if sum(total_predicted_list) else 1
                        
                        seq_attack_success_rate = sum(seq_human_deteced_all) / len(seq_human_deteced_all) if seq_human_deteced_all else 0

                        pbar.set_postfix({
                            'avg_attack_success_rate': f'{avg_attack_success_rate:.6f}',
                            'seq_attack_success_rate': f'{seq_attack_success_rate:.6f}'
                        })

            avg_attack_success_rate = 1-(sum(human_detected_list) / sum(total_predicted_list)) if sum(total_predicted_list) else 1
            logger.info('avg_attack_success_rate(higher better): {}'.format(avg_attack_success_rate))

            seq_attack_success_rate = sum(seq_human_deteced_all) / len(seq_human_deteced_all) if seq_human_deteced_all else 0
            logger.info('seq_attack_success_rate(higher better): {}'.format(seq_attack_success_rate))
            return seq_attack_success_rate
        

    def dist_validation(self, dataloader,current_iter, tb_logger, save_img,metrics):
        avg_attack_success_rate = 0
        seq_attack_success_rate =0
        
        # for avg_attack_success_rate
        total_predicted_list = []
        human_detected_list = []

        #for seq_attack_success_rate
        seq_human_deteced_all = []
        
        if "AP" in metrics:
            assert len(metrics)==1,"AP must be tested alone"
            with tqdm(total=len(dataloader), desc=f'Validate',leave=True) as pbar:
                for index, val_data in enumerate(dataloader):

                    self.feed_data(val_data)
                    rtval =self.test(output=True,metrics="AP")
                    pbar.update(1)  # 
            
            metrics , uv_texture = self.netG.cal_AP_after_epoch()
            AP = metrics['AP']
            AP_50 = metrics['AP_50']
            AP_75 = metrics['AP_75']
            AP_S = metrics['AP_S']
            AP_M = metrics['AP_M']
            AP_L = metrics['AP_L']
            logger.info('AP : {}, AP_50 : {}, AP_75 : {}, AP_S : {}, AP_M : {}, AP_L: {}'.format(AP,AP_50,AP_75,AP_S,AP_M,AP_L))
            return metrics,uv_texture
        else:
            with torch.no_grad():
                with tqdm(total=len(dataloader), desc=f'Validate',leave=True) as pbar:
                    for index, val_data in enumerate(dataloader):
                        self.feed_data(val_data)
                        for metric in metrics :
                            if metric == "avg_attack_success_rate":
                                rtval,uv_texture =self.test(output=True,metrics=metric)
                                human_detected, total_detected = rtval[0],rtval[1] 
                                #print(f"avg:  human_detected={human_detected}, total_detected={total_detected}")
                                total_predicted_list.append(total_detected)
                                human_detected_list.append(human_detected)
                                
                            if metric == "seq_attack_success_rate":
                                rtval,uv_texture =self.test(output=True,metrics=metric)
                                seq_human_detected,total_detected = rtval[0], rtval[1]
                                #print(f"seq:  human_detected={seq_human_detected}, total_detected={total_detected}")
                                seq_human_deteced_all.append(seq_human_detected)

                        pbar.update(1)  # 
                        
                        avg_attack_success_rate = 1-(sum(human_detected_list) / sum(total_predicted_list)) if sum(total_predicted_list) else 1
                        
                        seq_attack_success_rate = sum(seq_human_deteced_all) / len(seq_human_deteced_all) if seq_human_deteced_all else 0

                        pbar.set_postfix({
                            'avg_attack_success_rate': f'{avg_attack_success_rate:.6f}',
                            'seq_attack_success_rate': f'{seq_attack_success_rate:.6f}'
                        })

            avg_attack_success_rate = 1-(sum(human_detected_list) / sum(total_predicted_list)) if sum(total_predicted_list) else 1
            logger.info('avg_attack_success_rate(higher better): {}'.format(avg_attack_success_rate))

            seq_attack_success_rate = sum(seq_human_deteced_all) / len(seq_human_deteced_all) if seq_human_deteced_all else 0
            logger.info('seq_attack_success_rate(higher better): {}'.format(seq_attack_success_rate))
            return seq_attack_success_rate,uv_texture
    

    def feed_data(self, data):
        if ('verts_seq' in data.keys()) :
            self.verts_seq = data['verts_seq'].to(self.device)
        else:
            self.verts_seq = None

        if ('faces' in data.keys()) :
            self.faces = data['faces'].to(self.device)
        else:
            self.faces = None

        if ('t_val' in data.keys()) :
            self.T_val = data['t_val'].to(self.device) # control the render size of human 
        else:
            self.T_val = None

        if ('verts_uvs' in data.keys()) :
            self.verts_uvs = data['verts_uvs'].to(self.device)
        else:
            self.verts_uvs = None

        if ('faces_uvs' in data.keys()) :
            self.faces_uvs = data['faces_uvs'].to(self.device)
        else:
            self.faces_uvs = None

        if ('uv_input' in data.keys()) :
            self.uv_input = data['uv_input'].to(self.device)
        else:
            self.uv_input = None

        if ('labels' in data.keys()) :
            self.labels = data['labels'].to(self.device)
        else:
            self.labels = None

        if ('vis_mask' in data.keys()) :
            self.vis_mask = data['vis_mask'].to(self.device)
        else:
            self.vis_mask = None
                        
    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step,alpha):
        self.optimizer_G.zero_grad()

        object_confidence, class_confidence  = self.netG(self.verts_seq, self.faces, self.verts_uvs, self.faces_uvs, self.uv_input,self.labels,True,vis_mask=self.vis_mask,T_val = self.T_val,alpha=alpha)

        object_confidence *= self.lambda_w
        class_confidence *= self.lambda_w
        loss = object_confidence + class_confidence
        loss.backward()

        self.optimizer_G.step()

        self.log_dict['loss'] = loss.item()
        self.lrs.append(loss.item())
        self.log_dict['loss_mean'] = np.mean(self.lrs)

        self.log_dict['obj_conf'] = object_confidence.item()
        self.log_dict['cls_conf'] = class_confidence.item()
        self.obj_confs.append(object_confidence.item())
        self.cls_confs.append(class_confidence.item())
        self.log_dict['obj_mean'] = np.mean(self.obj_confs)
        self.log_dict['cls_mean'] = np.mean(self.cls_confs)

        torch.cuda.empty_cache()
   


    def test(self, output=False,metrics=None):
        self.netG.eval()
        with torch.no_grad():
            attack_success,uv_texture = self.netG(self.verts_seq, self.faces, self.verts_uvs, self.faces_uvs, self.uv_input,self.labels,metrics=metrics,vis_mask=self.vis_mask,T_val = self.T_val)
 
        self.netG.train()
        return attack_success,uv_texture



    def test_uvmap(self, uv_texture, output=False,metrics=None):
        self.netG.eval()
        with torch.no_grad():
            attack_success = self.netG.inference(self.verts_seq, self.faces, self.verts_uvs, self.faces_uvs, uv_texture,self.labels,metrics=metrics, vis_mask=self.vis_mask, T_val = self.T_val)
        self.netG.train()
        return attack_success
    

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['restore'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    
    # def validation(self, dataloader, current_iter, tb_logger,  save_img=False,metrics=None):
    #     """Validation function.
    #     Args:
    #         dataloader (torch.utils.data.DataLoader): Validation dataloader.
    #         current_iter (int): Current iteration.
    #         tb_logger (tensorboard logger): Tensorboard logger.
    #         save_img (bool): Whether to save images. Default: False.
    #     """
    #     seq_attack_success_rate = self.dist_validation(dataloader, current_iter, tb_logger, save_img,metrics)
    #     return seq_attack_success_rate
    def validation(self, dataloader, current_iter, tb_logger,  save_img=False,metrics=None):
        """Validation function.
        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        seq_attack_success_rate,uv_texture = self.dist_validation(dataloader, current_iter, tb_logger, save_img,metrics)
        return seq_attack_success_rate,uv_texture

    def inference_by_uvmap(self, dataloader,uv_texture, current_iter, tb_logger,  save_img=False,metrics=None):
        """Validation function.
        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """

        seq_attack_success_rate = self.dist_validation_uvmap(dataloader, uv_texture,current_iter, tb_logger, save_img,metrics)

        return seq_attack_success_rate