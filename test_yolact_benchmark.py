## A benchmark measuring speed for sequential calls to YOLACT inference (eval.py, InfTool.raw_inference())
# vs. using a batch mode.
from inference_tool import InfTool

import cv2
import numpy as np
import os

import timeit

IMGS='./data/yolact/datasets/dataset_kuka_env_pybullet_20/test/'
COUNT=1000
BATCH=1

if __name__ == '__main__':
  cnn = InfTool(weights='./data/yolact/weights/weights_yolact_kuka_17/crow_base_35_457142.pth', top_k=15, score_threshold=0.51, config='crow_base_config')
  
  #prepare data for work
  images = []
  for name in os.listdir(IMGS):
      img = cv2.imread(os.path.join(IMGS, name))
      images.append(img)
      if len(images)>= COUNT:
          break
  batches = [images[x:x+BATCH] for x in range(0, len(images), BATCH)]
  del images
  
  #process in batches and time it
  n=0
  t_start = timeit.default_timer()
  for batch in batches:
      assert len(batch)==BATCH

      #actual inference
      preds, frame = cnn.process_batch(batch, batchsize=BATCH)
      classes, scores, bboxes, masks = cnn.raw_inference(None, preds=preds, frame=frame, batch_idx=0)
      #print("processing batch {}".format(n))
      n+=1
  t_stop = timeit.default_timer()

  print("Running {} jobs with batch size {} took {} seconds.".format(COUNT, BATCH, t_stop-t_start))

  
