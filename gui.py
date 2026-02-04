import os
import sys
import time
from datetime import datetime
import math
from pathlib import Path
import numpy as np
import pandas as pd
import skimage as sk
from skimage import io as skio  # without this, the "scipy.special._cdflib" was not found by pyinstaller
from skimage import transform
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import filedialog, PhotoImage
import czifile
import multiprocessing as mp

from czi2tiff import czi2tiff
from nbt_intensity import nbt_intensity
from utils import resource_path, imread

use_cores = max(1, mp.cpu_count() - 2)
img_type = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp", ".czi")


def get_folder_path():
    global input_folder_dir
    f = filedialog.askdirectory()
    folder_path.set(f)
    input_folder_dir = folder_path.get()
    file_list = Path(input_folder_dir).rglob("*")
    img_list = [i for i in file_list if i.is_file() and i.suffix.lower() in img_type]
    img_list = [str(i).replace("\\", "/") for i in img_list]
    folder_img_num.set(f"From {input_folder_dir}\n" +
                       f"Load in {len(img_list)} images.")


def czi_to_tiff():
    #input_folder_dir = folder_path.get()
    output_folder_dir = f"{os.path.dirname(input_folder_dir)}/OUT_{os.path.basename(input_folder_dir)}"
    print(f"Input directory: {input_folder_dir}")
    print(f"Output directory: {output_folder_dir}")

    file_list = Path(input_folder_dir).rglob("*")
    img_list = [i for i in file_list if i.is_file() and i.suffix.lower() == ".czi"]
    img_list = [str(i).replace("\\", "/") for i in img_list]

    if len(img_list) == 0:
        error_message = f"From {input_folder_dir}\nNo images found."
        print(error_message)
        folder_img_num.set(error_message)
        return 0

    if not os.path.isdir(output_folder_dir):
        os.mkdir(output_folder_dir)
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start_time = time.perf_counter()
    ret = czi2tiff(img_list, input_folder_dir, output_folder_dir, use_cores)
    end_time = time.perf_counter()
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    done_message.set(
        f"--------- {round(end_time - start_time, 2)} secs ---------\n" +
        "Image conversion finished !"
    )
    print("Image conversion czi_to_tiff() finished.")


def run_nbt():
    #input_folder_dir = folder_path.get()
    output_folder_dir = f"{os.path.dirname(input_folder_dir)}/OUT_{os.path.basename(input_folder_dir)}"
    print(f"Input directory: {input_folder_dir}")
    print(f"Output directory: {output_folder_dir}")
    
    file_list = Path(input_folder_dir).rglob("*")
    img_list = [i for i in file_list if i.is_file() and i.suffix.lower() in img_type]
    img_list = [str(i).replace("\\", "/") for i in img_list]

    if len(img_list) == 0:
        folder_img_num.set(f"From {input_folder_dir}\nNo image found.")
        return 0

    if not os.path.isdir(output_folder_dir):
        os.mkdir(output_folder_dir)
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start_time = time.perf_counter()
    nbt_intensity(img_list, input_folder_dir, output_folder_dir, use_cores)
    end_time = time.perf_counter()
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    done_message.set(
        f"--------- {round(end_time - start_time, 2)} secs ---------\n" +
        "NBT measurement completed !"
    )
    print("NBT measurement run_nbt() completed.")


if __name__ == "__main__":
    mp.freeze_support()
    ############################################################################
    # Interface
    ############################################################################
    root = tk.Tk()
    root.title("NBT staining intensity")
    root.geometry("500x400")
    title_label = tk.Label(root, text="NBT intensity", font=("Calibri 24 bold")).pack(pady=5)
    #root.iconphoto(False, tk.PhotoImage(file = "icon.png"))
    root.iconbitmap(resource_path("icon.ico"))

    ############################################################################
    # Tk variables
    ############################################################################
    folder_path = tk.StringVar()
    folder_img_num = tk.StringVar(value = f"Current working directory:\n{os.getcwd()}")
    done_message = tk.StringVar()

    ############################################################################
    # Folder selection
    ############################################################################
    folder_frame = tk.Frame(master=root).pack(padx=20)

    folder_button = tk.Button(
        master=folder_frame,
        text="Select folder",
        font="Calibri 16",
        relief="solid",
        command=get_folder_path,
    ).pack(padx=20, pady=10, fill="x")

    folder_label = tk.Label(
        master=folder_frame,
        text="Select folder containing images",
        font="Calibri 15 italic",
        fg="gray50",
        textvariable=folder_img_num,
    ).pack(padx=20, pady=5, fill="x")

    ############################################################################
    # Convert .czi to .tiff
    ############################################################################
    tiff_frame = tk.Frame(master=root).pack(padx=20)

    tiff_button = tk.Button(
        master=tiff_frame,
        text="Convert .czi to .tiff",
        font="Calibri 16",
        relief= "ridge",
        cursor="exchange",
        command=czi_to_tiff,
    ).pack(padx=10, pady=5, anchor="center", fill="x")

    ############################################################################
    # Run NBT
    ############################################################################
    nbt_frame = tk.Frame(master=root).pack(padx=20)

    nbt_button = tk.Button(
        master=nbt_frame,
        text="Run NBT",
        font="Calibri 16",
        relief= "ridge",
        command=run_nbt,
    ).pack(padx=10, pady=10, anchor="center", fill="x")

    ############################################################################
    # Done message
    ############################################################################
    message_frame = tk.Frame(master=root).pack(padx=20)

    message_label = tk.Label(
        master=message_frame,
        text="---------",
        font="Calibri 15 italic",
        fg="gray50",
        textvariable=done_message,
    ).pack(padx=20, pady=5, fill="x")

    root.mainloop()
