import json
import sys, os

# when you're saving to json, make sure you convert dumps first then dump
def load_json(data_file):
    with open(data_file, 'r') as df:
        json_string = json.load(df)
        data = json.loads(json_string)
    return data


def save_json(json_object, data_file):
    with open(data_file, 'w') as df:
        json_object_string = json.dumps(json_object)
        json.dump(json_object_string, df)


def load_string_list(data_file):
    with open(data_file, 'r') as df:
        loaded_list = [line.strip() for line in df]

    return loaded_list


def save_string_list(obj_list, data_file):
    with open(data_file, 'w') as df:
        for s_object in obj_list:
            df.write('{:s}\n'.format(s_object))


def load_XYXY_list(data_file, n_item = 4):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        crops = []
        for s_line in lines:
            items = s_line.strip().split(',')
            assert len(items) == n_item, 'check file {:s} content'.format(data_file)
            s_crop = [int(items[0]), int(items[1]), int(items[2]), int(items[3])]
            crops.append(s_crop)
        return crops


def load_XYXYS_list(data_file, n_item = 5):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        crops = []
        for s_line in lines:
            items = s_line.strip().split(',')
            assert len(items) == n_item, 'check file {:s} content'.format(data_file)
            s_crop = [int(items[0]), int(items[1]), int(items[2]), int(items[3]), float(items[4])]
            crops.append(s_crop)
        return crops


def save_XYXY_list(obj_list, data_file):
    with open(data_file, 'w') as df:
        for s_object in obj_list:
            assert len(s_object)==4, 'Only accepts [x, y, x, y] format'
            df.write('{:d},{:d},{:d},{:d}\n'.format(*s_object))


def save_XYXYS_list(obj_list, data_file):
    with open(data_file, 'w') as df:
        for s_object in obj_list:
            assert len(s_object)==5, 'Only accepts [x, y, x, y, s] format'
            df.write('{:d},{:d},{:d},{:d},{:.4f}\n'.format(*s_object))


def save_XYXY_S_list(bbox_list, score_list, data_file):
    with open(data_file, 'w') as df:
        for s_bbox, s_score in zip(bbox_list, score_list):
                df.write('{:d},{:d},{:d},{:d},{:.4f}\n'.format(s_bbox[0], s_bbox[1],
                                                               s_bbox[2], s_bbox[3], s_score))


def split_XYXYS_list(bboxes_s_list):
    bboxes = []
    scores = []
    for s_bbox_s in bboxes_s_list:
        assert len(s_bbox_s) == 5, 'Only accepts [x, y, x, y, s] format'
        bboxes.append(s_bbox_s[0:4])
        scores.append(s_bbox_s[-1])
    return bboxes, scores


def merge_XYXY_S_list(bboxes, scores):
    bboxes_s = []
    for (s_bbox, s_score) in zip(bboxes, scores):
        s_bbox_s = s_bbox + s_score
        bboxes_s.append(s_bbox_s)

    return bboxes_s


def save_singlescore_list(score_list, save_file):
    """Saving list of scores, each element is saved to a row"""
    with open(save_file, 'wb') as f:
        for s in score_list:
            f.write('{:f}\r\n'.format(s))


def load_singlescore_list(annotation_file):
    """Loading a list of scores, each element is saved in a row"""
    with open(annotation_file, 'rb') as f:
        lines = f.read()
        elements = lines.strip().split('\n')
        elements = [float(x.strip()) for x in elements]
        return elements