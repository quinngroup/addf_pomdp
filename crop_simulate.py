# Driver packages
import numpy as np
from numpy.linalg import norm
import sys
import glob
import re
from random import randint as rand

# Image packages
import cv2

class CropField:
    def __init__(self, image_path):
        self.sectors = rand(2,4)
        # TODO: Process an image to generate simulation for bootstrapping
        # image = cv2.imread(image_path,0)
        # self.dimensions = image.shape[:2]