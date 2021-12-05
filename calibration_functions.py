import json
import numpy as np
from PIL import Image

"""
all functions expect to get image path of the form "{camera_id}_*.{png/npy}"

"""
def flatfield_correct(image_path):
    if image_path.endswith("png"):
        image = np.array(Image.open(image_path))
    elif image_path.endswith("npy"):
        image = np.load(image_path)
    else:
        print("image format not recognized")
    cam_id = image_path.split("_")[0]
    with open(f"{cam_id}_calibration_params.json", "r") as file:
        param_dict = json.load(file)
        cover_fix = np.array(param_dict["fixer image"])
        dark_cover = np.array(param_dict["I_dark"])
    
    return (image - dark_cover)*cover_fix


"""
dark_cover needs to have equal exposure time as the new given image.
"""
def convert_radince(image_path, exposure_time = 3000):
    if image_path.endswith("png"):
        image = np.array(Image.open(image_path))
    elif image_path.endswith("npy"):
        image = np.load(image_path)
    else:
        print("image format not recognized")
    cam_id = image_path.split("_")[0]
    with open(f"{cam_id}_calibration_params.json", "r") as file:
        param_dict = json.load(file)
        calibration_factor = np.array(param_dict["C"])
        cover_fix = np.array(param_dict["fixer image"])
        dark_cover = np.array(param_dict["I_dark"])
        
    return cover_fix*((image-dark_cover)/(calibration_factor*exposure_time))

