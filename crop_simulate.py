# Driver packages
import numpy as np
from numpy.linalg import norm
import sys
import glob
import re

# Image packages
import cv2

class CropField:
    def __init__(self, image_path):
        image = cv2.imread(image_path,0)
        self.size = image.shape[:2]