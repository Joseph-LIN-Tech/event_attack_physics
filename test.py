import os
from NetWorks.modules.Attack_RVT import attack_rvt
from Options.trainOptions import TrainOptions
import hydra #import main
from omegaconf import DictConfig, OmegaConf
import random
import logging
import option
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
import option
from torch.utils.data import DataLoader
from utils import util
from NetWorks import create_model
from detection import Module as rnn_det_module
from datetime import datetime
import math
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
        # if phase == 'train':
        #     transform = MyTransform(args=dataset_opt, reshape_size=(240,304),crop_size=dataset_opt['size'],debug_mode=dataset_opt['debug'], train_mode=True)
        #     dataset_train = PoseDataSet(dataset_opt, transform, debug_mode=dataset_opt['debug'], train_mode=True,phase=phase)
        #     train_loader = DataLoader(dataset_train, num_workers=dataset_opt['num_workers'], batch_size=dataset_opt['batch_size'], shuffle=True)
            
        #     train_size = int(math.ceil(len(dataset_train) / dataset_opt['batch_size']))
        #     total_iters = int(args['train']['niter'])
        #     total_epochs = int(math.ceil(total_iters / train_size))
        #     logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
        #             len(dataset_train), train_size))
        #     logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
        #             total_epochs, total_iters))
        #     print("len(self.train_loader)",len(train_loader))
        if phase == 'val':    
            transform = MyTransform(args=dataset_opt, reshape_size=(240,304),crop_size=dataset_opt['size'],debug_mode=dataset_opt['debug'], train_mode=False)
            dataset_val = PoseDataSet(dataset_opt, transform, debug_mode=dataset_opt['debug'], train_mode=False,phase=phase)
            val_loader = DataLoader(dataset_val, num_workers=dataset_opt['num_workers'], batch_size=dataset_opt['batch_size'], shuffle=False)
            
            val_metrics = dataset_opt['metrics']

            logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(dataset_val)))
            
        #### resume training
    # if resume_state:
    #     logger.info('Resuming training from epoch: {}, iter: {}.'.format(
    #         resume_state['epoch'], resume_state['iter']))

    #     start_epoch = resume_state['epoch']
    #     current_step = resume_state['iter']
    #     model.resume_training(resume_state)  # handle optimizers and schedulers
    # else:
    current_step = 0
    #start_epoch = 0

   
    model.validation(val_loader, current_step, tb_logger=None,save_img=args['val']['save_img'],metrics=val_metrics)
    model.save(current_step)
    logs = model.get_current_log()
    message = "" 
    for v in model.get_current_learning_rate():
        message += '{:.3e},'.format(v)
    message += ')  '
        
    logger.info(message)

          

 
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
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='Options/test.yaml', help='Path to option YAML file.')
    args = parser.parse_args()
    args = option.parse(args.opt, is_train=True)
    # with open('Options/train.yaml', 'r') as file:
    #     args = yaml.safe_load(file)
    with open('Options/rvt_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # convert to OmegaConf's DictConfig object
    config = OmegaConf.create(config)
    main(args,config)
