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
from eval import prep_display
from data.config import Config, set_cfg

class InfTool:

    def __init__(self,
                 weights='./data/yolact/weights/weights_yolact_kuka_17/crow_base_35_457142.pth',
                 config=None,
                 top_k=25,
                 score_threshold=0.1
                 ): 
        # initialize a yolact net for inference
        # print or override params here, see crow_yolact_wrapper.py for options

        ## YOLACT setup
        # setup config
        if config is not None:
          if '.obj' in config:
              with open(config,'rb') as f:
                  config = dill.load(f)
          set_cfg(config)
        # setup yolact args #TODO there's a nicer way: yolact.eval.parse_args('.....')
        global args
        args=Config({})
        args.top_k = top_k
        args.score_threshold = score_threshold
        # set here everything that would have been set by parsing arguments in yolact/eval.py:
        args.display_lincomb = False
        args.crop = False
        args.display_fps = False
        args.display_text = True
        args.display_bboxes = True
        args.display_masks =True
        args.display_scores = True

        # CUDA setup for yolact
        torch.backends.cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        #YOLACT net itself
        net = Yolact().cuda()
        net.load_weights(weights)
        net.eval()
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False

        self.net = net
        print("YOLACT network available as self.net")
        self.args = args


    def process_batch(self, img):
        """
        To speed up processing (avoids duplication if label_image & raw_inference is used)
        """
        frame = torch.from_numpy(img).cuda().float() #TODO how to make frame/batch with multiple images at once?
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        return preds, frame


    def label_image(self, img, preds=None, frame=None):
        """
        optional args preds, frame: obtained from process_batch(). Can be used to speedup cached inferences.
        """
        if preds is None or frame is None:
          preds, frame = self.process_batch(img)
        global args
        processed = prep_display(preds, frame, h=None, w=None, undo_transform=False, args=args)
        return processed


    def raw_inference(self, img, preds=None):
        """
        optional arg preds: if not None, avoids process_batch() call, used to speedup cached inferences.
        """
        if preds is None:
          preds, _ = self.process_batch(img)
        global args
        w,h,_ = img.shape
        [classes, scores, boxes, masks] = postprocess(preds, w=w, h=h, batch_idx=0, interpolation_mode='bilinear', 
                                                      visualize_lincomb=False, crop_masks=True, score_threshold=args.score_threshold)
        #TODO do we want to keep tensor, or convert to py list[]?
        return [classes, scores, boxes, masks] #TODO also compute and return centroids?

