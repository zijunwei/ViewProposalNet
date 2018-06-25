import os
import sys
sys.path.append((os.path.dirname(__file__)))
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import Network
flags = tf.app.flags
from py_utils import bboxes as boxes
import project_utils
import datasets.proposal_data_transforms as transforms
from py_utils import file_utils, load_utils
from tf_utils import utils as tf_utils
# import progressbar
import glob


def main(argv=None):
    gpu_id = 0
    test_image_directory = './example_images'
    image_format = 'jpg'
    image_list = glob.glob(os.path.join(test_image_directory, '*.{:s}'.format(image_format)))
    image_list.sort()
    result_save_directory = 'ProposalResults'
    save_file = os.path.join(result_save_directory, 'ViewProposalResults-tmp.txt')
    anchors = project_utils.get_pdefined_anchors(anchor_file='datasets/pdefined_anchor.pkl')
    model_weight_path = './pretrained/ProposalNet/VPN'

    # Machine Learning Cores:
    data_transform = transforms.get_val_transform(image_size=320)


    with tf.Graph().as_default():
        # not using batch yet
        with tf.variable_scope('inputs'):
            tf_image = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='image_input')

        p_logits, _, _, _ = Network.base_net(tf_image, num_classes=len(anchors), rois=None, is_training=False, bbox_regression=False)

        init_fn = slim.assign_from_checkpoint_fn(
            model_weight_path,
            slim.get_model_variables(), ignore_missing_vars=True)  # set to true to avoid the incompatibal ones


        config = tf_utils.gpu_config(gpu_id=gpu_id)
        image_annotation = {}

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            # pbar = progressbar.ProgressBar(max_value=len(image_list))
            for id, s_image_path in enumerate(image_list):
                    s_image_name = os.path.basename(s_image_path)
                    # pbar.update(id)
                    s_image = file_utils.default_image_loader(s_image_path)
                    s_image_width, s_image_height = s_image.size

                    s_image_tensor = data_transform(s_image)
                    s_image_tensor = s_image_tensor.unsqueeze(0)
                    s_image_np = np.transpose(s_image_tensor.numpy() * 256, [0, 2, 3, 1])

                    rpn_ = sess.run(p_logits,
                                 feed_dict={tf_image: s_image_np})
                    logits = rpn_[0]
                    s_bboxes = []

                    for anchor_idx, s_anchor in enumerate(anchors):
                        s_bbox = s_anchor[0:4]

                        s_bboxes.append([int(s_bbox[0]*s_image_width), int(s_bbox[1]*s_image_height),
                                         int(s_bbox[2]*s_image_width), int(s_bbox[3]*s_image_height)])

                    scores_selected, bboxes_selected = boxes.sortNMSBBoxes(logits, s_bboxes, NMS_thres=0.6)#TODO: This is a tunable parameter

                    # TODO: only keep Top5
                    topN = 5
                    pick_n = min(topN, len(scores_selected))
                    image_annotation[s_image_name] = {}
                    image_annotation[s_image_name]['scores'] = scores_selected[0:pick_n]
                    image_annotation[s_image_name]['bboxes'] = bboxes_selected[0:pick_n]
        print "Done Computing, saving to {:s}".format(save_file)
        load_utils.save_json(image_annotation, save_file)
if __name__ == '__main__':
    tf.app.run()