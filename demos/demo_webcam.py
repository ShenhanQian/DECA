# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors
from skimage.transform import estimate_transform, warp


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

def img2crop(image, face_detector, crop_size=224, scale=1.25):
    h, w, _ = image.shape

    # provide kpt as txt file, or mat file (for AFLW2000)
    bbox, bbox_type = face_detector.run(image)
    if len(bbox) < 4:
        print('no face detected! run original image')
        left = 0; right = h-1; top=0; bottom=w-1
    else:
        left = bbox[0]; right=bbox[2]
        top = bbox[1]; bottom=bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)

    size = int(old_size*scale)
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

    DST_PTS = np.array([[0,0], [0,crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    image = image/255.

    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2,0,1)
    return {'image': torch.tensor(dst_image).float(),
            'imagename': 'webcam',
            'tform': torch.tensor(tform.params).float(),
            'original_image': torch.tensor(image.transpose(2,0,1)).float(),
            }

def main(args):
    device = args.device

    face_detector = detectors.FAN()

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
  
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    
    while(True): 
        # Capture the video frame 
        # by frame 
        ret, image = vid.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = img2crop(image, face_detector)
        images = data['image'].to(device)[None,...]

        with torch.no_grad():
            codedict = deca.encode(images, use_detail=False)

            tform = data['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = data['original_image'][None, ...].to(device)
            if args.vis:
                opdict, visdict = deca.decode(codedict, render_orig=True, original_image=original_image, use_detail=False, tform=tform)

                visdict['inputs'] = original_image
                # vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                # vis_name = 'landmarks2d'
                vis_name = 'shape_images'
                image = util.tensor2image(visdict[vis_name][0])
            else:
                opdict = deca.decode(codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False, render_orig=False)
    
        # Display the resulting frame 
        cv2.imshow('frame', image) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--vis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize of output' )
    main(parser.parse_args())