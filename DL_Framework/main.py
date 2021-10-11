import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os
import warnings
import sys
from launcher import *
from utils.Initialization import *
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings('ignore')

if __name__=='__main__':
    np.random.seed(12345)
    params = process_config('./config/config_segmentation_3d.cfg')
    operation = Launcher(params)
    if params['is_training'] == True:
        op = operation.train()
    else:
        op = operation.test()