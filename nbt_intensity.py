import os
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

from utils import imread

date_time = datetime.now().strftime("%Y%m%d%H%M%S")

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Internal function, calculate NBT measures only for a single image
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def _nbt_intensity(img_path: str, input_folder_path: str, output_folder_path: str):
    img_dirname = os.path.dirname(img_path)
    img_basename = os.path.basename(img_path)
    contour_output_path = os.path.join(
        img_dirname.replace(input_folder_path, output_folder_path),
        img_basename.replace(Path(img_basename).suffix, "_NBT.jpg")
    )
    try:
        if not os.path.exists(os.path.dirname(contour_output_path)):
            os.makedirs(os.path.dirname(contour_output_path), exist_ok = True)

        RGB, GRAY, img_shape = imread(img_path)

        ###########################################################################################################
        # If the image is blank
        ###########################################################################################################
        if np.std(GRAY) == 0:
            GRAY = Image.fromarray(GRAY)
            GRAY.save(contour_output_path, format = "JPEG")
            return img_dirname, img_basename, 0, 0, 0, "cancel"
        
        ###########################################################################################################
        # Generate masking area
        ###########################################################################################################
        mask = transform.resize(GRAY, output_shape = (512, 512), anti_aliasing = True, preserve_range = True)
        mask = np.uint8(mask)
        kernel_size = np.sum(mask > 0) / (512 * 512) * 100 * 1.1
        kernel_size = math.ceil(kernel_size)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # coerce to odd value
        
        ## Eliminate noises ====
        neighborhood = sk.morphology.disk(kernel_size)
        for _ in range(5):
            mask = sk.filters.rank.median(mask, neighborhood)

        ## Otsu threshold ====
        try:
            threshold = sk.filters.threshold_multiotsu(image = mask, classes = 5)
        except:
            threshold = [0, 0, 0, 0, 255]

        mask = mask > np.median(threshold)
        mask = transform.resize(mask, output_shape = GRAY.shape, anti_aliasing = False, preserve_range = True)
        
        #Image.fromarray(mask).save(contour_output_path, format = "JPEG", quality = 95)

        ## Erosion + Dilation ==== 

        ###########################################################################################################
        # Extract region of interest (ROI)
        ###########################################################################################################
        ROI = np.multiply(mask, GRAY)

        ###########################################################################################################
        # Output NBT measure parameters
        ###########################################################################################################
        nbt_total_intensity = np.sum(ROI)
        nbt_area = np.sum(mask)  # How many pixels
        nbt_avg_intensity = nbt_total_intensity / nbt_area if nbt_area > 0 else 0

        ###########################################################################################################
        # Generate contour
        ###########################################################################################################
        contours = sk.measure.find_contours(mask, level = 0)
        RGB = Image.fromarray(RGB)
        draw = ImageDraw.Draw(RGB)
        contour_color = (255, 0, 0) if len(img_shape) == 3 else 255
        for contour in contours:
            points = [(c[1], c[0]) for c in contour]    # (x, y)
            draw.line(points, fill = contour_color, width = 10)
        font_settings = ImageFont.load_default(size=90)
        draw.text(
            xy = (30, 10), 
            text = f"Avg: {round(nbt_avg_intensity, 2)} = {round(nbt_total_intensity/1_000_000, 2)} M / {nbt_area} pixels",
            fill = contour_color,
            font = font_settings
        )
        RGB.save(contour_output_path, format = "JPEG", quality = 95)
        return img_dirname, img_basename, nbt_total_intensity, nbt_area, nbt_avg_intensity, ""
    except:
        print(img_path, "cancel")
        return img_dirname, img_basename, -999, -999, -999, "cancel"


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Process multiple images
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def nbt_intensity(img_list: str, input_folder_path: str, output_folder_path: str, use_cores: int):
    img_num = len(img_list)
    use_cores = min(use_cores, img_num)
    df_output = {
        "img_dirname": [],
        "img_basename": [],
        "nbt_total_intensity": [],
        "nbt_area": [],
        "nbt_avg_intensity": [],
        "note": []
    }

    if img_num < 10 or use_cores < 2:
        # Single process ====
        for img in img_list:
            print(f"Processing: {img}")
            ret = _nbt_intensity(img, input_folder_path, output_folder_path)
            for k, v in zip(df_output.keys(), ret):
                df_output[k].append(v)
    else:
        # Multi-process
        print(f"Multiprocesssing...\nUse {use_cores} cores.")
        tasks = [(i, input_folder_path, output_folder_path) for i in img_list]
        pool = mp.Pool(use_cores)
        ret = pool.starmap(_nbt_intensity, tasks)  # return the value as a tuple
        pool.close()
        pool.join()
        for val in ret:
            for k, v in zip(df_output.keys(), val):
                df_output[k].append(v)

    df_output = pd.DataFrame.from_dict(df_output)
    df_output.to_csv(f"{output_folder_path}/OUT_NBT_{date_time}.csv", index = False)
    return 0
