import math
from py_utils import load_utils
# generate candidate crops following:
# https://arxiv.org/pdf/1701.01480.pdf
# with the following difference:
# this is no longer 5 * 5 grid, instead we first of all find the smaller size offset, and make sure the step is larger than 10 pixel
# and slide over
def generateCandidateCrops_M(image_size, include_orig=True):
    h, w = image_size
    min_gridsize = 5
    if include_orig:
        scales = [x / 10. for x in range(5, 11)]
    else:
        scales = [x / 10. for x in range(5, 10)]

    image_aspect_ratio = w * 1./ h
    aspect_ratios = [1., 3./2, 3./4, 4./3, 2./3, 16./9, 9./16, image_aspect_ratio]

    #round off aspect ratios to 2 digits
    aspect_ratios = [ round(i * 100)  * 1. / 100 for i in aspect_ratios]
    # aspect_ratios = list(set(aspect_ratios))

    xyxys = []
    for s_scale in scales:
        for s_aspect_ratio in aspect_ratios:
            scaled_size = math.sqrt(w * h) * s_scale
            new_w = int(scaled_size * math.sqrt(s_aspect_ratio))
            new_h = int(scaled_size / math.sqrt(s_aspect_ratio))
            if new_h > h or new_w > w:
                continue

            offset_x = ( (w - new_w) / (min_gridsize - 1))
            offset_y = ( (h - new_h) / (min_gridsize - 1))

            #incase there is no offsest at all. if this is 1, then there might be situations that too many
            min_offset = max(20, min(offset_x, offset_y))

            gridsize_x = int(math.floor((w - new_w) / min_offset) + 2)
            gridsize_y = int(math.floor((h - new_h) / min_offset) + 2)

            for idx_x in range(gridsize_x):
                for idx_y in range(gridsize_y):
                    # new_xyxy = [min_offset*idx_x, min_offset*idx_y, min(w, min_offset*idx_x + new_w), min(h, min_offset*idx_y + new_h)]
                    # not need for the check
                    # new_xyxy = [min_offset*idx_x, min_offset*idx_y,  min_offset*idx_x + new_w, min_offset*idx_y + new_h]
                    new_xyxy = [min_offset*idx_x, min_offset*idx_y, min(w, min_offset*idx_x + new_w), min(h, min_offset*idx_y + new_h)]

                    xyxys.append(new_xyxy)

    return xyxys

def generateCandidateCropsForEval(image_size, scales=None, include_orig=True):
    h, w = image_size
    min_gridsize = 5
    if scales is None:
        if include_orig:
            scales = [x / 10. for x in range(5, 11)]
        else:
            scales = [x / 10. for x in range(5, 10)]

    image_aspect_ratio = w * 1./ h
    aspect_ratios = [1., 3./2, 3./4, 4./3, 2./3, 16./9, 9./16, image_aspect_ratio]

    #round off aspect ratios to 2 digits
    aspect_ratios = [round(i * 100) * 1. / 100 for i in aspect_ratios]
    aspect_ratios = list(set(aspect_ratios))

    xyxys = []
    for s_scale in scales:
        for s_aspect_ratio in aspect_ratios:
            scaled_size = math.sqrt(w * h) * s_scale
            new_w = int(scaled_size * math.sqrt(s_aspect_ratio))
            new_h = int(scaled_size / math.sqrt(s_aspect_ratio))
            if new_h > h or new_w > w:
                continue

            offset_x = ( (w - new_w) / (min_gridsize - 1))
            offset_y = ( (h - new_h) / (min_gridsize - 1))

            #incase there is no offsest at all. if this is 1, then there might be situations that too many
            min_offset = max(20, min(offset_x, offset_y))

            gridsize_x = int(math.floor((w - new_w) / min_offset) + 2)
            gridsize_y = int(math.floor((h - new_h) / min_offset) + 2)

            for idx_x in range(gridsize_x):
                for idx_y in range(gridsize_y):
                    new_xyxy = [min_offset*idx_x, min_offset*idx_y, min(w, min_offset*idx_x + new_w), min(h, min_offset*idx_y + new_h)]
                    xyxys.append(new_xyxy)

    return xyxys


if __name__ == '__main__':
    BBOXes = generateCandidateCropsForEval([450, 800], scales=[0.7, 0.8,0.9, 1.0])
    # load_utils.save_num_list(BBOXes, 'debug.txt')
    # LoadedBBoxes = load_utils.load_XYXYS_list('debug.txt')
    print "Done"