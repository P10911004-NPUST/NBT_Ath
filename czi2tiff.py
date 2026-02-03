import os
from pathlib import Path
from PIL import Image
import numpy as np
import czifile
import multiprocessing as mp

def _czi2tiff(czi_path: str, input_folder_path: str, output_folder_path: str):
    '''Arguments definition
    czi_path: the absolute directory of the single .czi image file
    input_folder path: Since the input accepts only a folder and recursively searches for .czi files in
        the child directory, the parent dirname was required to replace the directory of each tiff files.
        For example, if input_folder_path = "C:/a/b", and one of the czi image is "C:/a/b/c/d/e.czi", 
        then the output RGB tiff file should be "C:/a/OUT_b/c/d/e_RGB.tiff".
    output_folder_path: The reason is same with the input_folder_path argument.
    '''
    if (Path(czi_path).suffix != ".czi"):
        raise ValueError("No .czi file was found.")
    
    czi_dirname = os.path.dirname(czi_path)
    czi_basename = os.path.basename(czi_path)
    RGB_output_path = os.path.join(
        czi_dirname.replace(input_folder_path, output_folder_path), 
        "OUT_RGB",
        czi_basename.replace(Path(czi_basename).suffix, "_RGB.tiff")
    )
    GRAY_output_path = os.path.join(
        czi_dirname.replace(input_folder_path, output_folder_path),
        "OUT_GRAY",
        czi_basename.replace(Path(czi_basename).suffix, "_GRAY.tiff")
    )
    
    czi = czifile.imread(czi_path)
    img = np.array(czi, dtype = np.double)
    _, width, height, channel = img.shape
    RGB = img[0, :, :, :]
    #R = RGB[:, :, 0]
    #G = RGB[:, :, 1]
    #B = RGB[:, :, 2]
    #RGB = np.stack([R, G, B], axis = -1)
    #RGB = (RGB - RGB.min()) / (RGB.max() - RGB.min()) * 255.0
    #RGB = RGB.astype(np.uint8)
    RGB = np.divide(RGB, RGB.max()) * 255.0
    RGB = RGB.astype(np.uint8)
    RGB = Image.fromarray(RGB, mode = "RGB")  # Takes only uint8 data type
    if not os.path.exists(os.path.dirname(RGB_output_path)):
        os.makedirs(os.path.dirname(RGB_output_path), exist_ok = True)
    RGB.save(RGB_output_path, format = "TIFF")

    GRAY = np.mean(RGB, axis = 2)
    #GRAY = (GRAY - GRAY.min()) / (GRAY.max() - GRAY.min())
    #GRAY = (1.0 - GRAY) * 255.0
    #GRAY = GRAY.astype(np.uint8)
    GRAY = GRAY.max() - GRAY
    GRAY = Image.fromarray(GRAY)
    if not os.path.exists(os.path.dirname(GRAY_output_path)):
        os.makedirs(os.path.dirname(GRAY_output_path), exist_ok = True)
    GRAY.save(GRAY_output_path, format = "TIFF")
    return 0


def czi2tiff(img_list: str, input_folder_path: str, output_folder_path: str, use_cores: int):
    img_num = len(img_list)
    use_cores = min(use_cores, img_num)
    # Single process
    if img_num < 10:
        for img in img_list:
            print(f"Processing: {img}")
            ret = _czi2tiff(img, input_folder_path, output_folder_path)
    else:
    # Multi-process
        print(f"Multiprocesssing...\nUse {use_cores} cores.")
        tasks = [(i, input_folder_path, output_folder_path) for i in img_list]
        pool = mp.Pool(use_cores)
        ret = pool.starmap(_czi2tiff, tasks)
        pool.close()
        pool.join()
    return 0