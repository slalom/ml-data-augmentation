import cv2
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser(description="Generate additional dataset using combination of rotation / flip horizontal / gamma adjustment")
parser.add_argument('--image-folder', type=str, required=True, help='Folder containing images to generate variations on')
args = parser.parse_args()

filedir = args.image_folder

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def rotate_180(image):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
    
for full_path in glob.glob(filedir + "/*.jpg"):
    
    g_dark = 0.75
    g_light = 1.25
    
    basename = os.path.basename(full_path)
    filename, fileext = os.path.splitext(basename)
    
    image = cv2.imread(full_path)
    print("Original JPG: " + full_path)
    orig_dark = adjust_gamma(image, g_dark)
    orig_light = adjust_gamma(image, g_light)
    rotated = rotate_180(image)
    rotated_dark = adjust_gamma(rotated, g_dark)
    rotated_light = adjust_gamma(rotated, g_light)
    
    #flip horizontal
    fliph = cv2.flip(image, 0)
    fliph_dark = adjust_gamma(fliph, g_dark)
    fliph_light = adjust_gamma(fliph, g_light)
    fliph_rot = rotate_180(fliph)
    fliph_rot_dark = adjust_gamma(fliph_rot, g_dark)
    fliph_rot_light = adjust_gamma(fliph_rot, g_light)
    
    newfilebase = filedir + "/" + filename
    orig_dark_filename = newfilebase + "_dark" + fileext
    orig_light_filename = newfilebase + "_light" + fileext
    rotated_filename = newfilebase + "_rotated" + fileext
    rotated_dark_filename = newfilebase + "_rotated_dark" + fileext
    rotated_light_filename = newfilebase + "_rotated_light" + fileext
    
    fliph_filename = newfilebase + "_fliph" + fileext
    fliph_dark_filename = newfilebase + "_fliph_dark" + fileext
    fliph_light_filename = newfilebase + "_fliph_light" + fileext
    fliph_rot_filename = newfilebase + "_fliph_rot" + fileext
    fliph_rot_dark_filename = newfilebase + "_fliph_rot_dark" + fileext
    fliph_rot_light_filename = newfilebase + "_fliph_rot_light" + fileext
    
    print("Original Dark: " + orig_dark_filename)
    cv2.imwrite(orig_dark_filename, orig_dark)
    print("Original Light: " + orig_light_filename)
    cv2.imwrite(orig_light_filename, orig_light)
    print("Rotated: " + rotated_filename)
    cv2.imwrite(rotated_filename, rotated)
    print("Rotated and Dark: " + rotated_dark_filename)
    cv2.imwrite(rotated_dark_filename, rotated_dark)
    print("Rotated and Light: " + rotated_light_filename)
    cv2.imwrite(rotated_light_filename, rotated_light)
    
    print("Flipped Horizontal: " + fliph_filename)
    cv2.imwrite(fliph_filename, fliph)
    print("Flipped Horizontal and Dark: " + fliph_dark_filename)
    cv2.imwrite(fliph_dark_filename, fliph_dark)
    print("Flipped Horizontal and Light: " + fliph_light_filename)
    cv2.imwrite(fliph_light_filename, fliph_light)
    print("Flipped Horizontal and Rotated: " + fliph_rot_filename)
    cv2.imwrite(fliph_rot_filename, fliph_rot)
    print("Flipped Horizontal and Rotated and Dark: " + fliph_rot_dark_filename)
    cv2.imwrite(fliph_rot_dark_filename, fliph_rot_dark)
    print("Flipped Horizontal and Rotated and Light: " + fliph_rot_light_filename)
    cv2.imwrite(fliph_rot_light_filename, fliph_rot_light)
    
    print("-------------------------")
    
print("***COMPLETE***")