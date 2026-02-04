import os
import sys
from datetime import datetime
from pathlib import Path
import math
import numpy as np
import pandas as pd
import skimage as sk
from skimage import transform
from PIL import Image, ImageDraw, ImageFont
import multiprocessing as mp
import czifile

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Read image
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def imread(img_path):
    img_type = Path(img_path).suffix.lower()
    if img_type == ".czi":
        img = czifile.imread(img_path)
        img = np.array(img, dtype = np.double)
        _, width, height, channel = img.shape
        img = img[0, :, :, :]
        RGB = np.divide(img, img.max()) * 255.0
        RGB = RGB.astype(np.uint8)
        GRAY = np.mean(img, axis = 2)
    if img_type in (".tiff", ".tif", ".jpg", ".jpeg", ".png", ".bmp"):
        img = sk.io.imread(img_path)
        if len(img.shape) == 2:
            '''the img shape is (height, width)'''
            #RGB = np.stack([img, img, img], axis = -1, dtype = np.uint8)
            RGB = img
            RGB = RGB / RGB.max() * 255.0 if RGB.max() > 0 else RGB
            RGB = np.uint8(RGB)
            GRAY = np.double(img)
            return RGB, GRAY, img.shape
        if len(img.shape) == 3:
            '''the img shape is (height, width, channel)'''
            RGB = img
            GRAY = sk.color.rgb2gray(img)
            GRAY = np.double(GRAY)
    GRAY = GRAY / GRAY.max() if GRAY.max() > 0 else GRAY
    GRAY = (GRAY.max() - GRAY) * 255.0  # invert foreground and background
    '''RGB return as uint8, GRAY return as double'''
    return RGB, GRAY, img.shape

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Get the icon path during runtime
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def resource_path(relative_path):
    """ Get absolute path to resource (works for PyInstaller) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)