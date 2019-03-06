import os
import sys
import tarfile
import argparse

import cv2
import numpy

parser = argparse.ArgumentParser('Extract background images from a tar archive.')
parser.add_argument('--archive', required=True)
parser.add_argument('--folder-name', required=True)

args=parser.parse_args()

def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR)


def extract_backgrounds(archive, folder):
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    t = tarfile.open(name=archive)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f =  t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue
        
        if im.shape[0] > im.shape[1]:
            #im = im[:im.shape[1], :]
            continue
        else:
            im = im[:, :im.shape[0]]
            if im.shape[0] > 640 and im.shape[0] < 1200:
                im = cv2.resize(im, (640, 640))
                fname = "{}/{:08}.jpg".format(folder,index)
                print(fname)
                rc = cv2.imwrite(fname, im)
                if not rc:
                    raise Exception("Failed to write file {}".format(fname))
                index += 1


if __name__ == "__main__":
    extract_backgrounds(args.archive, args.folder_name)
