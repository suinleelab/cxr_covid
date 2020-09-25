#!/usr/bin/env python
# make_h5.py
#
import argparse
import h5py
import io
import numpy
import os
import torchvision.transforms
import tqdm
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import datasets.padchestdataset
PADCHEST_CORRUPTED = set(datasets.padchestdataset.CORRUPTED)

def is_image(path):
    lower = path.lower()
    if lower.startswith('.') or lower.endswith('fake.jpg'):
        return False
    if lower.endswith('.jpg') or lower.endswith('png'):
        return True
    else:
        return False

def find_images(parentpath):
    '''
    Find images in ``parentpath`` and all subdirectories. Return list of paths 
    (strings).
    '''
    paths = []
    for triple in os.walk(parentpath, followlinks=True):
        for path in triple[-1]:
            if is_image(path):
                paths.append(os.path.join(triple[0], path))
    # remove path prefix
    for ipath, path in enumerate(paths):
        paths[ipath] = path[len(parentpath):]
    return paths

def save_png_bytes(h5handle, path, image):
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='PNG')
    image_bytes = image_byte_array.getvalue()
    # magic
    image_raw_np = numpy.asarray(image_bytes)
    h5handle['images'].create_dataset(path, data=image_raw_np)

def convert_dataset():
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Scale(224),
            torchvision.transforms.CenterCrop(224)
            ])

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='imagedir')
    parser.add_argument("-o", dest='outpath')
    args = parser.parse_args()

    h5handle = h5py.File(args.outpath, 'w', libver='latest')
    h5handle.swmr_mode = True

    h5handle.create_group("images")

    # directory structure is <main directory>/<image names>
    print("\n")
    img_paths = find_images(args.imagedir)
    for ip in tqdm.tqdm(img_paths):
        if not os.path.basename(ip) in PADCHEST_CORRUPTED:
            path = os.path.join(os.path.abspath(args.imagedir), ip.strip('/')) 
            image = Image.open(path)
            # downsample
            image = transform(image)
            save_png_bytes(h5handle, ip, image)

if __name__ == "__main__":
    convert_dataset()
