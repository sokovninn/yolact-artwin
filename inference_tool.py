# This class works as a convenience wrapper over the YOLACT
# functionality, allowing you to import and call yolact INFERENCE
# conveniently in an object-oriented manner. 
#
# Existing and working YOLACT installation is necessary. 

import sys
import os
import torch
import cv2
import numpy as np
import pkg_resources
import dill

# import YOLACT
sys.path.append(os.path.relpath("."))
from yolact import Yolact
from data import set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from eval import prep_display, parse_args
from data.config import set_cfg

import timeit


class InfTool:

    def __init__(self,
                 weights='./data/yolact/weights/weights_yolact_kuka_17/crow_base_35_457142.pth',
                 config=None,
                 top_k=25,
                 score_threshold=0.1,
                 display_text=True,
                 display_bboxes=True,
                 display_masks=True,
                 display_scores=True,
                 ): 
        self.score_threshold = score_threshold
        self.top_k = top_k
        
        # initialize a yolact net for inference
        ## YOLACT setup
        # setup config
        if config is not None:
          if '.obj' in config:
              with open(config,'rb') as f:
                  config = dill.load(f)
          set_cfg(config)

        parse_args(['--top_k='+str(top_k), 
                    '--score_threshold='+str(score_threshold),
                    '--display_text='+str(display_text),
                    '--display_bboxes='+str(display_bboxes),
                    '--display_masks='+str(display_masks),
                    '--display_scores='+str(display_scores),])

        # CUDA setup for yolact
        torch.backends.cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        #YOLACT net itself
        with torch.no_grad():
            net = Yolact().cuda()
            net.load_weights(weights)
            net.eval()
            net.detect.use_fast_nms = True
            net.detect.use_cross_class_nms = False

        self.net = net
        print("YOLACT network available as self.net")
        self.args = args

        #for debug,benchmark
        self.duration=0.0


    def process_batch(self, img, batchsize=1):
        """
        To speed up processing (avoids duplication if label_image & raw_inference is used)
        """
        if not isinstance(img, list):
            img = [img]
            
        start = timeit.default_timer()
        imgs = np.stack(img, axis=0)
        imgs = np.asarray(imgs, dtype=np.float32)
        stop = timeit.default_timer()
        self.duration+=(stop-start)

        with torch.no_grad():
            frame = torch.from_numpy(imgs)
            frame = frame.cuda().float()
            batch = FastBaseTransform()(frame)
            preds = self.net(batch)
        return preds, frame


    def label_image(self, img, preds=None, frame=None):
        """
        optional args preds, frame: obtained from process_batch(). Can be used to speedup cached inferences.
        """
        if preds is None or frame is None:
          preds, frame = self.process_batch(img)
        processed = prep_display(preds, frame, h=None, w=None, undo_transform=False)
        return processed


    def raw_inference(self, img, preds=None, frame=None, batch_idx=None):
        """
        optional arg preds, frame: if not None, avoids process_batch() call, used to speedup cached inferences.
        """
        if preds is None or frame is None:
          preds, frame = self.process_batch(img)
        n,w,h,_ = frame.shape
        if n > 1:
            assert batch_idx is not None, "In batch mode, you must provide batch_idx - meaning which row of batch is used as the results, [0, {}-1]".format(n)
        [classes, scores, boxes, masks] = postprocess(preds, w=w, h=h, batch_idx=batch_idx, interpolation_mode='bilinear', 
                                                      visualize_lincomb=False, crop_masks=True, score_threshold=self.score_threshold)
        #TODO do we want to keep tensor, or convert to py list[]?
        return [classes, scores, boxes, masks] #TODO also compute and return centroids?

