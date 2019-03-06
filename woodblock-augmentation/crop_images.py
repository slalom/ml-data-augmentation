import cv2
import numpy as np
import argparse
import glob
import os

import rawpy
import imageio

parser = argparse.ArgumentParser(description="Convert RAW images to JPG and crop images based on contrast")
parser.add_argument('--raw-image-folder', type=str, required=True, help='RAW image folder location')
args = parser.parse_args()

filedir = args.raw_image_folder

if not os.path.exists(filedir+'/jpg'):
    os.makedirs(filedir+'/jpg')
    
if not os.path.exists(filedir+'/jpg/cropped'):
    os.makedirs(filedir+'/jpg/cropped')
    
jpgdir = filedir+'/jpg'
croppeddir = jpgdir + '/cropped'

# Convert raw images CR2 in png
for full_path in glob.glob(filedir + "/*.CR2"): 
    filename = os.path.basename(full_path)
    print(full_path)
    raw = rawpy.imread(full_path)
    image = raw.postprocess()
    imageio.imsave(jpgdir + "/" + filename.split(".")[0] + ".jpg", image)   


for full_path in glob.glob(jpgdir + "/*.jpg"): 

    filename = os.path.basename(full_path)
    # Original image
    image = cv2.imread(full_path)
    print(full_path)
    # # Resize
    draw = np.zeros_like(image)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get black and white image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # Some erosions and dilations to remove noise.
    thresh = cv2.erode(thresh, kernel, iterations=4)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    
    # Get Contours of binary image
    im, cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the biggest contour
    max_area = -1
    max_c = 0
    for i in range(len(cnts)):
        contour = cnts[i]
        area = cv2.contourArea(contour)
        if (area > max_area):
            max_area = area
            max_c = i
    contour = cnts[max_c]
    
    # Get minAreaRect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # write results
    
    W = rect[1][0]
    H = rect[1][1]
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    
    angle = rect[2]
    if angle < -45:
        angle += 90
    
    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
    
    cv2.imwrite(croppeddir + "/" + filename, croppedRotated)
    
