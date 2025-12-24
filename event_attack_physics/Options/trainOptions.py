#-*- coding:utf-8 -*-
import argparse
import os

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--model', default="generator", help='model to train,options:fcn8,segnet...')  
        #self.parser.add_argument('--model-dir', default="./Model/TrainingModels/16/generator_94.pth", help='path to stored-model')   #204c fenzhi
        #self.parser.add_argument('--model-dir', default="./Model/generatorIJRR.pth", help='path to stored-model')   #204c fenzhi
        self.parser.add_argument('--model-dir', default="Model/generatorMVSEC.pth", help='path to stored-model')   
        self.parser.add_argument('--data-simu', default="/home/lgx/data/whitebox_attack/train/",help='path where image.txt lies')
        self.parser.add_argument('--data-simu-pose', default="/home/lgx/data/whitebox_attack_cmupose/train/",help='path where pose.npz lies')
        self.parser.add_argument('--reshape-size', default=(240,304), help='resize the image')
        self.parser.add_argument('--bins', default=10, help='the number of bins')
        self.parser.add_argument('--gpu-ids', default=[0,1], help='the index of gpus')
        self.parser.add_argument('--seq_len', default=3, help='the length of event sequences')
        self.parser.add_argument('--smpl-model-dir',default="body_models",help="the model of the SMPL and DMPL")
        self.parser.add_argument('--uv_obj_path',default="/home/lgx/code/ECCV2024/blackbox_attack/event_attack_human_GA_0220_uv_allbody/support_data/uv_template_obj/smpl_uv.obj",help="the model of the SMPL and DMPL")
        self.parser.add_argument('--size', default=(240,304), help='crop the image')
        #self.parser.add_argument('--save-dir', type=str, default='Results/',help='options. visualize the result of segmented picture, not just show IoU')
        self.parser.add_argument('--debug', action='store_true', default=False) # debug mode 
        self.parser.add_argument('--dl', type=int, default=0, help='crop the image')
        self.parser.add_argument('--dr', type=int, default=281, help='crop the image')
        self.parser.add_argument('--data-size', type=int, default=20,help='path where image.txt lies')
        self.parser.add_argument('--save-dir', type=str,default='./Model/TrainingModels/23/',help='savedir for models')#old20201126
        self.parser.add_argument('--tb-path', type=str,default='./Model/TrainingModels/23/',help='savedir for tensorboardX')#old20201126
        self.parser.add_argument('--output-rgb-path', type=str,default='output_data/RGB',help='savedir for tensorboardX')#old20201126
        
        self.parser.add_argument('--contrast_threshold_pos', type=float, default=0.1)
        self.parser.add_argument('--contrast_threshold_neg', type=float, default=0.1)        
        self.parser.add_argument('--g-lr', type=float, default=4e-4)
        self.parser.add_argument('--d-lr', type=float, default=4e-4)
        self.parser.add_argument('--train-g-interval', type=int, default=5)
        self.parser.add_argument('--num-epochs', type=int, default=10)
        self.parser.add_argument('--num-workers', type=int, default=0)
        self.parser.add_argument('--batch-size', type=int, default=1)
        self.parser.add_argument('--d-batch-size', type=int, default=256)
        self.parser.add_argument('--epoch-save', type=int, default=5)    #You can use this value to save model every X epochs
        self.parser.add_argument('--show-interval', type=int, default=1)
        self.parser.add_argument('--steps-loss', type=int, default=20)
        self.parser.add_argument('--warmup', type=int, default=0)
        self.parser.add_argument('--lr-end', type=int, default=1e-5)
        self.parser.add_argument('--trainstep', type=int, default=100)
        
        self.parser.add_argument('--minimum-threshold', type=float, default=0.01)
        
        self.parser.add_argument("--patch", type=int, default=4, help="image patch size")

        self.parser.add_argument("--use-log-images", default=True, help="irradiance maps = log(gray map)")
        self.parser.add_argument("--log-eps", type=int, default=0.001, help="E = log(L/255+eps)")
        self.parser.add_argument("--log-threshold", type=int, default=20, help="x>=T E = log(L) or x<T E = L/Tlog(T)")
        
        self.parser.add_argument("--clamp-num", type=float, default=0.01, help="WGAN clip gradient")
        self.parser.add_argument("--gp-lambda", type=float, default=1, help="WGAN gradient penalty")
        self.parser.add_argument("--min-event-num", type=float, default=10, help="avoid all black")
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
