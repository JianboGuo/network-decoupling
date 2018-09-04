from __future__ import print_function
from easydict import EasyDict as edict
import config as cfgs
import os
import argparse
import os.path as osp
import pickle
import sys
from multiprocessing import Process, Queue

import numpy as np

from decompose import *
from net import Net
from utils import *
from worker import Worker
import google.protobuf.text_format

#TODO: offer API and test function

def decomp_step1(pt,weight):
    #preprocess
    net = Net(pt=pt,model=weight)
    WPQ, pt, model = net.preprocess()
    return {'WPQ':WPQ,'pt':pt,'model':model}

def decomp_step2(pt, weight, WPQ, check_exist=False, **kwargs):
    net = Net(pt=pt, model=weight)
    model = net.finalmodel(WPQ) # loads weights into the caffemodel

    if 'CD_param' in kwargs:
        data = kwargs['CD_param']['enable']
    else:
        data = cfgs.CD_param['enable']

    if not data:
        if 'SD_param' in kwargs:
            data = kwargs['SD_param']['data_driven']
        else:
            data = cfgs.SD_param['data_driven']

    if 'mask_layers' in kwargs:
        mask = kwargs['mask_layers']
    else:
        mask = cfgs.mask_layers

    if data:
        sample_convs = [conv for conv in net.type2names()]
        if net.resnet:
            sample_convs += net.type2names('Eltwise')
        net.freeze_images(check_exist=check_exist, convs=sample_convs)
        return {'WPQ':WPQ, 'pt':mem_pt(pt), 'model':model}
    else:
        return {'WPQ':WPQ, 'pt':pt, 'model':model}

def decomp_step3(pt,weight,**kwargs):
    #decouple
    net = Net(pt=pt,model=weight,**kwargs)
    WPQ, pt = net.decompose()
    return {'WPQ':WPQ, 'pt':pt, 'weight':weight}

def decomp_step4(pt,weight,WPQ,**kwargs):
    # load weight
    net = Net(pt=pt,model=weight)
    new_model = net.finalmodel(WPQ=WPQ, save=False)
    if 'CD_param' in kwargs:
        data = kwargs['CD_param']['enable']
    else:
        data = cfgs.CD_param['enable']

    if not data:
        if 'SD_param' in kwargs:
            data = kwargs['SD_param']['data_driven']
        else:
            data = cfgs.SD_param['data_driven']
            
    net.dis_memory()
    new_pt = net.seperateReLU()
    new_pt, new_model = net.save(prefix='new')
     
    return {'new_pt':new_pt, 'new_model':new_model}

def decomp(pt, weight, **kwargs):
    """
    do decomposition given prototxt, weight and other params
    param:
        pt: the prototxt to define original model
        weight: the caffemodel file name of original model
        kwargs: other params as follows
            ND_param: EasyDict object, contain parameters of network deoupling
            SD_param: EasyDict object, contain parameters of spatial decomposition
            CD_param: EasyDict object, contain parameters of channel decomposition
            device_id: int, determine the device number (CPU or GPU) to use
            nSamples: int, number of samples when using data-driven approach
            nPointsPerSample: int, number of points sampled per sample when using data-driven approach
            accname: string, the accuracy(or output) layer name of the network
            frozen_name: string, the name of pickle file saving sampled features
            mask_layers: list object, names of some convolution layers that do not need decouple
    """
    wk = Worker()
    printstage('preprocessing network')
    out = wk.do(target=decomp_step1,pt=pt, weight=weight)
    # sample for data driven method
    printstage('sample data for reconstruction')
    out = wk.do(target=decomp_step2,pt=out['pt'], weight=out['model'],WPQ=out['WPQ'],**kwargs)
    printstage('decoupling and saving weight results')
    out = wk.do(target=decomp_step3,pt=out['pt'], weight=out['model'],**kwargs)
    printstage('loading new model and save')
    result = wk.do(target=decomp_step4, **out, **kwargs)
    print('save decouple results:')
    print(result['new_pt'])
    print(result['new_model'])
    #print('result flops')
    #compute(pt=result['new_pt'],model=result['new_model'])

def compute(pt, model):
    net = Net(pt=pt, model=model)
    net.computation()

def caffe_test(pt, model, iterations=200 , gpu=None, time=False):
    if gpu is None:
        gpu = cfgs.device_id
    shell(cfgs.caffe_path, 'test', '-gpu', gpu, '-weights', model, '-model', pt , '-iterations', str(iterations))
    if time:
        shell(cfgs.caffe_path, 'time', '-gpu', gpu, '-weights', model, '-model', pt)

def parse_args():
    parser = argparse.ArgumentParser("decouple CNN")
    parser.add_argument('-sd',dest='sd',help='enable spatial decomposition',action='store_true')
    parser.add_argument('-nd',dest='nd',help='enable network decoupling',action='store_true')
    parser.add_argument('-cd',dest='cd',help='enable channel decompostion',action='store_true')
    parser.add_argument('-data',dest='data',help='enable data driven for spatial decomposition',action='store_true')
    parser.add_argument('-speed', dest='speed', help='sd speed up ratio of spatial decomposition', default=None, type=float)
    parser.add_argument('-threshold', dest='threshold', help='energy threshold for network decouple',default=None,type=float)
    parser.add_argument('-gpu', dest='gpu', help='caffe devices', default=None, type=int)
    parser.add_argument('-model', dest='model', help='caffe prototxt file path', default=None, type=str)
    parser.add_argument('-weight', dest='weight', help='caffemodel file path', default=None, type=str)
    parser.add_argument('-action', dest='action', help='compute, test or decompose the model', default='decomp',\
     type=str,choices=['decomp','compute','test'])
    parser.add_argument('-iterations', dest='iter', help='test iterations', type=int, default=1)
    parser.add_argument('-rank',dest='rank',help='rank for network decoupling', type=int, default=0)
    parser.add_argument('-DP', dest='DP', help='flags to set DW + PW decouple (default is PW + DW)', action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    DEBUG = 0
    if args.action == 'decomp':
        ND_param = edict()
        SD_param = edict()
        CD_param = edict()
        CD_param['c_ratio'] = args.speed
        CD_param['enable'] = args.cd
        SD_param['c_ratio'] = args.speed
        SD_param['enable'] = args.sd
        SD_param['data_driven'] = args.data
        ND_param['enable'] = args.nd
        ND_param['energy_threshold'] = args.threshold
        ND_param['rank'] = args.rank
        ND_param['DP'] = args.DP
        mask_layers = cfgs.mask_layers
        decomp(pt=args.model,weight=args.weight,SD_param=SD_param, ND_param=ND_param, CD_param=CD_param
            ,gpu=args.gpu, mask_layers=mask_layers)
    elif args.action == 'compute':
        compute(pt=args.model,model=args.weight)
    elif args.action == 'test':
        caffe_test(pt=args.model,model=args.weight, gpu=args.gpu, iterations=args.iter)
    else:
        NotImplementedError
