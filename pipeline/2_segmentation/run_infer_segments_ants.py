import cv2
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import numpy as np
from sourceRdefs import getFilterImageIDs
from inferSegments import inferSegments

# get image IDs not in training set, not already classified
# image_ids = getFilterImageIDs(exclude_training=False)

image_ids = [str(x) for x in [1,2,3]] #,3,4,5,6,7,8,9]]

# infer segments
inferSegments(image_ids=image_ids,model_location="G:/ant-patterner/data/ml_models/ant_head_segment.pt",image_size=200,show=True, proj_dir="G:/ant-patterner")

