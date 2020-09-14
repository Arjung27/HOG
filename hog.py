import time
import torch
import numpy as np

from PIL import Image
from skimage.feature import hog

class HOGTransform(object):
    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(3, 3), stacked_dims=False, gpu=-1):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.stacked_dims = stacked_dims
        self.gpu = gpu

    def __call__(self, img):
        begin = time.time()
        if self.gpu == 0:
            print('start of calling hog transfor')
        np_img = np.array(img)
        #
        hog_feature = hog(np_img, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, 
                          visualize=False, feature_vector=False, multichannel=True)  # shape [H/ppc, W/ppc, cpb, cpb, 9]
        if self.gpu == 0:
            print('skimage hog transfor: ', time.time() - begin)
        begin = time.time()
        H, W = hog_feature.shape[:2]
        if self.stacked_dims:
            output = np.reshape(hog_feature, (H, W, -1))
        else:
            p = self.cells_per_block[0]
            output = np.reshape(hog_feature, (H*p, W*p, -1))
        if self.gpu == 0:
            print('numpy reshape: ', time.time() - begin)
        begin = time.time()
        output = np.transpose(output, (2, 0, 1))
        output = torch.from_numpy(output).float()
        if self.gpu == 0:
            print('torch tensor: ', time.time() - begin)
        return output
        

