import os, sys
import glob
from PIL import Image
import progressbar

def get_file_list(image_dir, file_format='jpg'):
    file_list = glob.glob(os.path.join(image_dir, '*.{:s}'.format(file_format)))
    return file_list


def default_image_loader(path):
    return Image.open(path).convert('RGB')

def verify_image_list(image_list, attemptOpen=False, image_loader=default_image_loader, verbose=False):
    verified_image_list = []
    if verbose:
        pbar = progressbar.ProgressBar(max_value=len(image_list))

    for i, s_image_path in enumerate(image_list):
        if verbose:
            pbar.update(i)
        if os.path.isfile(s_image_path):
            if attemptOpen:
                try:
                    image_loader(s_image_path)
                    verified_image_list.append(s_image_path)
                except:
                    print "Cannot Load {:s}".format(s_image_path)
            else:
                verified_image_list.append(s_image_path)
        else:
            print "Image Not Exist: {:s}".format(s_image_path)
    return verified_image_list