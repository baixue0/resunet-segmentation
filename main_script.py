import sys

PY_LIB_PATH = r'/mnt/d/DDrive/pylib'
sys.path.insert(0, PY_LIB_PATH)

import os
import numpy as np
import SimpleITK as sitk

from dlseg.segmentation2d import Segmentation2D

cwd = os.getcwd()

def readmhd(mhdfilename):
    sitkimage = sitk.ReadImage(os.path.join(cwd,mhdfilename))
    spacing = sitkimage.GetSpacing()
    ndarr = sitk.GetArrayFromImage(sitkimage).astype(np.int32)
    return spacing, ndarr

def writemhd(spacing, ndarr, writepath):
    sitkimage = sitk.GetImageFromArray(ndarr)
    sitkimage.SetSpacing(spacing)
    sitk.WriteImage(sitkimage, writepath, useCompression=True)

def fill_bk(arr):
    '''
    replace background pixels (value 0) with mean value of non-background pixel
    '''
    bkmsk = arr==0
    arr[bkmsk] = arr[~bkmsk].mean()
    return arr

if __name__ == "__main__":
    output_dir = os.path.join(cwd, sys.argv[0].split('.')[0])
    print(output_dir)

    # prepare traning pairs
    train_gray_2dlist, train_label_2dlist = [],[]
    
    _, gray_arr = readmhd(r'D3459-3slices.mhd')
    _, label_arr = readmhd(r'D3459-3slices_edited.mhd')
    train_slices = np.arange(3)# use all 3 slices of the stack
    train_gray_2dlist += [gray_arr[i] for i in train_slices]
    train_label_2dlist += [label_arr[i] for i in train_slices]

    _, gray_arr = readmhd(r'D3459-30slices.mhd')
    _, label_arr = readmhd(r'D3459-30slices_edited.mhd')
    train_slices = np.arange(8)# use first 8 slices of the stack
    gray_arr = fill_bk(gray_arr)
    train_gray_2dlist += [gray_arr[i] for i in train_slices]
    train_label_2dlist += [label_arr[i] for i in train_slices]

    print('train slices shape: ',[ele.shape for ele in train_gray_2dlist])

    seg2d = Segmentation2D(
        directory=output_dir,
        device='cuda:0',
    )

    seg2d.new_model(
        name='model0', 
        network='resunet', 
        num_classes=label_arr.max() + 1, 
        tile_size=512
    )

    seg2d.ndarray_to_tensor(train_gray_2dlist, train_label_2dlist, train_gray_2dlist, train_label_2dlist)

    seg2d.train(
        num_epochs=100,
        steps_per_epoch=100,
        batch_size=8,
    )

    # inference
    spacing, infer_arr = readmhd(r'D3459.mhd')
    path_predict_label = os.path.join(output_dir,r'D3459-dlseg.mhd')
    infer_arr = fill_bk(infer_arr)
    infer_arr = np.stack(seg2d.infer_2dlist(infer_arr, batch_size=8, showprogress=True))
    writemhd(spacing, infer_arr, path_predict_label)
