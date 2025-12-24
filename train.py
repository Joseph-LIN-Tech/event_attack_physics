import os
from NetWorks.modules.Attack_RVT import attack_rvt
from Options.trainOptions import TrainOptions
import hydra #import main
from omegaconf import DictConfig, OmegaConf
import random
import logging
# import argparse
import debugpy
# debugpy.listen(5679)
# debugpy.wait_for_client()

from callbacks_rvt.custom import get_ckpt_callback, get_viz_callback
from callbacks_rvt.gradflow import GradFlowLogCallback
from config_rvt.modifier import dynamically_modify_train_config
from data_rvt.utils.types import DatasetSamplingMode
from loggers_rvt.utils import get_wandb_logger, get_ckpt_path
from modules_rvt.utils.fetch import fetch_data_module, fetch_model_module
from pathlib import Path
from DataLoader.transform import MyTransform
from DataLoader.dataset import MyDataSet,PoseDataSet
import torch
from torch.backends import cuda, cudnn
print("torch.version.cuda",torch.version.cuda) 
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.loggers import CSVLogger

from torch.utils.data import DataLoader
from utils import util
from NetWorks import create_model
from detection import Module as rnn_det_module
from datetime import datetime
import math
import option
def main(args,config):
    '''
        Train the model and record training options.
    '''
    rank = -1
    # config logger
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'output/experiment/training_{current_time}.log'
    logging.basicConfig( level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logger = logging.getLogger()
     #### loading resume state if exists
    if args['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(args['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(args, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    seed = 0
    if seed is None:
        seed = random.randint(1, 10000)

    util.set_random_seed(seed)
        
    model = create_model(args,config)
    model_params = util.get_model_total_params(model)
    print("model_params",model_params)
    logger.info('cell_size: {:d} '.format(args['network']['cell_size']))
    for phase, dataset_opt in args['datasets'].items():
        if phase == 'train':
            transform = MyTransform(args=dataset_opt, reshape_size=(240,304),crop_size=dataset_opt['size'],debug_mode=dataset_opt['debug'], train_mode=True)
            dataset_train = PoseDataSet(dataset_opt, transform, debug_mode=dataset_opt['debug'], train_mode=True,phase=phase)
            train_loader = DataLoader(dataset_train, num_workers=dataset_opt['num_workers'], batch_size=dataset_opt['batch_size'], shuffle=True)
            
            train_size = int(math.ceil(len(dataset_train) / dataset_opt['batch_size']))
            total_iters = int(args['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(dataset_train), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
            
        elif phase == 'val':    
            transform = MyTransform(args=dataset_opt, reshape_size=(240,304),crop_size=dataset_opt['size'],debug_mode=dataset_opt['debug'], train_mode=False)
            dataset_val = PoseDataSet(dataset_opt, transform, debug_mode=dataset_opt['debug'], train_mode=False,phase=phase)
            val_loader = DataLoader(dataset_val, num_workers=dataset_opt['num_workers'], batch_size=dataset_opt['batch_size'], shuffle=False)
            
            val_metrics = dataset_opt['metrics']

            logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(dataset_val)))
       


     #train data loader
    print("len(self.train_loader)",len(train_loader))

    print("========== TRAINING ===========")
    start_epoch = 0

    best_seq_attack_success_rate = 0
    current_step = 0
    alpha = 0
    for epoch in range(start_epoch, total_epochs+1):
        for iter_id, train_data in enumerate(train_loader):
            model.feed_data(train_data)
            alpha = epoch/(total_epochs-1) 
            
            model.optimize_parameters(current_step,alpha)
            model.update_learning_rate(current_step, warmup_iter=-1)
            
            ### pring the losses 
            if current_step % args['logger']['print_freq'] == 0 and current_step>0:
                logs = model.get_current_log()
                message = '  epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')  '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    
                if rank <= 0:
                    logger.info(message)

            #### save models and training states
            if (current_step % args['logger']['save_checkpoint_freq'] == 0 or (current_step == total_iters)) and current_step>0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
                    
                    seq_attack_success_rate,uv_texture = model.validation(val_loader, current_step, tb_logger=None,save_img=args['val']['save_img'],metrics=val_metrics)
                    model.save_uv_texture(uv_texture,current_step)
                    logs = model.get_current_log()
                    message = "" 
                    for v in model.get_current_learning_rate():
                        message += '{:.3e},'.format(v)
                    message += ')  '

                    #save the best model
                    if seq_attack_success_rate >= best_seq_attack_success_rate:
                        model.save(f"best_{current_step}")
                        model.save_training_state(epoch, current_step)
                        model.save_uv_texture(uv_texture,"best_"+str(current_step))
                        best_seq_attack_success_rate = seq_attack_success_rate

                    
            if (args['val'] is not None) and (current_step % args['val']['val_freq'] == 0) and current_step>0:
                if rank <= 0:
                    if current_step % args['logger']['save_checkpoint_freq'] == 0:#don't need to do again
                        pass
                    else:
                        #model.save_uv_texture(attacked_uv_texture,current_step)
                        seq_attack_success_rate,uv_texture = model.validation(val_loader, current_step, tb_logger=None,save_img=args['val']['save_img'],metrics=val_metrics)
                        logs = model.get_current_log()
                        message = "" 
                        for v in model.get_current_learning_rate():
                            message += '{:.3e},'.format(v)
                        message += ')  '
                        for k, v in logs.items():
                            if k=="accuracy" or k=="attack_success_rate":
                                message += '{:s}: {:.4e} '.format(k, v)
                            
                        
                        logger.info(message)

            current_step += 1

            if current_step > total_iters:
                break
        # if current_step > total_iters:
        #     break
    print("========== TRAINING FINISHED ===========")

 
@hydra.main(config_path='config_rvt', config_name='val', version_base='1.2')
def rvt_model_init(config: DictConfig) -> rnn_det_module:
    global rvt_model 
    logger = CSVLogger(save_dir='./validation_logs')
    print("config",config)

    ckpt_path = Path(config.checkpoint)
    dynamically_modify_train_config(config)
    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    rvt_model = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})


import yaml

if __name__ == '__main__':
    with open('Options/train.yaml', 'r') as file:
        args = yaml.safe_load(file)
    with open('Options/rvt_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # convert to OmegaConf's DictConfig object
    config = OmegaConf.create(config)
    main(args,config)
