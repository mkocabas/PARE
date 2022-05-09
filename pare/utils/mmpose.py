# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import json
import copy
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_top_down_pose_model, init_pose_model, vis_pose_result
)

from pare.core.config import (
    MMPOSE_PATH, MMDET_PATH, MMPOSE_CKPT, MMPOSE_CFG, MMDET_CFG, MMDET_CKPT
)


def run_mmpose_with_dets(image_file_names, detections, show_results=False, results_folder=None):

    device = 'cuda:0'
    bbox_thr = 0.3
    kpt_thr = 0.3

    if results_folder:
        os.makedirs(results_folder, exist_ok=True)
    # det_model = init_detector(MMDET_CFG, MMDET_CKPT, device=device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(MMPOSE_CFG, MMPOSE_CKPT, device=device)

    dataset = pose_model.cfg.data['test']['type']
    dets = copy.deepcopy(detections)
    joints2d = []
    for img_idx, img_fname in enumerate(image_file_names):
        # test a single image, the resulting box is c_x, c_y, w, h
        person_bboxes = dets[img_idx]
        person_bboxes[:, 0] = person_bboxes[:, 0] - (person_bboxes[:, 2] // 2)
        person_bboxes[:, 1] = person_bboxes[:, 1] - (person_bboxes[:, 3] // 2)
        person_bboxes = np.concatenate([person_bboxes, np.ones((person_bboxes.shape[0], 1))], axis=-1)

        # test a single image, with a list of bboxes.
        pose_results = inference_top_down_pose_model(
            pose_model,
            img_fname,
            person_bboxes,
            bbox_thr=bbox_thr,
            format='xywh',
            dataset=dataset)

        # show the results
        vis_pose_result(
            pose_model,
            img_fname,
            pose_results,
            dataset=dataset,
            kpt_score_thr=kpt_thr,
            show=show_results,
            out_file=None if results_folder else os.path.join(results_folder, img_fname.split('/')[-1]),
        )

        if len(pose_results) < 1:
            joints2d.append(np.zeros((133, 3)))
        else:
            joints2d.append(pose_results[0]['keypoints'])

    return np.array(joints2d)


def run_mmpose(image_folder, show_results=False):


    device = 'cuda:0'
    bbox_thr = 0.3
    kpt_thr = 0.3


    det_model = init_detector(MMDET_CFG, MMDET_CKPT, device=device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(MMPOSE_CFG, MMPOSE_CKPT, device=device)

    dataset = pose_model.cfg.data['test']['type']

    # image_name = os.path.join(args.img_root, args.img)
    image_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
    ])

    joints2d = []
    for image_name in tqdm(image_names):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        det_results = inference_detector(det_model, image_name)

        # keep the person class bounding boxes. (FasterRCNN)
        person_bboxes = det_results[0].copy()
        # import IPython; IPython.embed(); exit()
        # test a single image, with a list of bboxes.
        pose_results = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_bboxes,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset)

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            kpt_score_thr=kpt_thr,
            show=show_results,
            out_file=None
        )

        if len(pose_results) < 1:
            joints2d.append(np.zeros((133, 3)))
        else:
            joints2d.append(pose_results[0]['keypoints'])

    return np.array(joints2d)

