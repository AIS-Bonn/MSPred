"""
SynpickVP dataset. Obtaining video sequences from the SynpickVP dataset, along with
the corresponding semantic segmentation maps and gripper locations.
"""

import json
import math
import os
import imageio
import torch
from torchvision import transforms
import numpy as np
from collections import defaultdict
from CONFIG import CONFIG, METRIC_SETS
from data.heatmaps import HeatmapGenerator
from .base_dataset import SequenceDataset


class SynpickMoving(SequenceDataset):
    """
    Each sequence depicts a robotic suction cap gripper that moves around in a red bin filled with objects.
    Over the course of the sequence, the robot approaches 4 waypoints that are randomly chosen from the 4 corners.
    On its way, the robot is pushing around the objects in the bin.
    """

    CATEGORIES = ["master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle",
                  "tuna_fish_can", "pudding_box", "gelatin_box", "potted_meat_can", "banana", "pitcher_base",
                  "bleach_cleanser", "bowl", "mug", "power_drill", "wood_block", "scissors", "large_marker",
                  "large_clamp", "extra_large_clamp", "foam_brick", "gripper"]
    NUM_CLASSES = len(CATEGORIES)  # 22
    NUM_HMAP_CHANNELS = [NUM_CLASSES + 1, 1]
    STRUCT_TYPE = ["SEGMENTATION_MAPS", "KEYPOINT_BLOBS"]
    BIN_SIZE = (373, 615)
    SKIP_FIRST_N = 72            # To skip the first few frames in which gripper is not visible.
    GRIPPER_VALID_OFFSET = 0.01  # To skip sequences where the gripper_pos is near the edges of the bin.

    METRICS_LEVEL_0 = METRIC_SETS["video_prediction"]
    METRICS_LEVEL_1 = METRIC_SETS["segmentation"]
    METRICS_LEVEL_2 = METRIC_SETS["single_keypoint_metric"]

    def __init__(self, split, num_frames, seq_step=2, img_size=(136, 240), hmap_size=(64, 112)):
        """ Dataset initializer """
        assert split in ["train", "val", "test"]
        data_path = CONFIG["paths"]["data_path"]
        self.data_dir = os.path.join(data_path, "SYNPICK", split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Synpick dataset does not exist in {self.data_dir}...")
        self.split = split
        self.img_size = img_size
        self.hmap_size = hmap_size
        self.num_frames = num_frames
        self.seq_step = seq_step

        # obtaining paths to data
        images_dir = os.path.join(self.data_dir, "rgb")
        scene_dir = os.path.join(self.data_dir, "scene_gt")
        masks_dir = os.path.join(self.data_dir, "masks")
        self.image_ids = sorted(os.listdir(images_dir))
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.scene_gt_fps = [os.path.join(scene_dir, scene) for scene in sorted(os.listdir(scene_dir))]
        self.mask_fps = [os.path.join(masks_dir, mask_fp) for mask_fp in sorted(os.listdir(masks_dir))]

        # linking data into a structure
        self.object_poses = {}
        self._create_frame_to_obj_kpoints_dict()
        self.valid_idx = []
        self.allow_seq_overlap = (split != "test")
        self._find_valid_sequences()
        return

    def __len__(self):
        """ """
        return len(self.valid_idx)

    def __getitem__(self, i):
        """ Sampling sequence from the dataset """
        i = self.valid_idx[i]  # only consider valid indices
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        idx = range(i, i + seq_len, self.seq_step)  # create range of indices for frame sequence
        imgs = [imageio.imread(self.image_fps[id_]) / 255. for id_ in idx]

        # preprocessing
        imgs = np.stack(imgs, axis=0)
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
        imgs = transforms.Resize(self.img_size)(imgs)
        # segmentation maps
        maps_1 = self._get_segmentation_maps(idx, num_classes=self.NUM_HMAP_CHANNELS[0])
        maps_1 = torch.Tensor(maps_1)
        # center-keypoint heatmaps
        # maps_2 = self._get_kpoint_heatmaps(idx, categories=range(1, self.NUM_CLASSES+1))
        maps_2 = self._get_kpoint_heatmaps(idx, categories=[self.NUM_CLASSES])
        maps_2 = torch.Tensor(maps_2)

        data = {"frames": imgs, "heatmaps": [maps_1, maps_2]}
        return data

    def _get_segmentation_maps(self, idx, num_classes):
        """ Loading segmentation maps for sequence idx """
        # masks = [imageio.imread(self.mask_fps[id_]) for id_ in idx]
        masks = np.array([imageio.imread(self.mask_fps[id_]) for id_ in idx])
        masks = torch.from_numpy(masks)
        masks = transforms.Resize(
                size=self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )(masks)
        seg_maps = [np.zeros((num_classes, *self.hmap_size), dtype=np.int8) for i in range(len(masks))]
        one_hot_mat = np.eye(num_classes, dtype=np.int8)
        for f in range(len(masks)):
            for i in range(self.hmap_size[0]):
                for j in range(self.hmap_size[1]):
                    seg_maps[f][:, i, j] = one_hot_mat[int(masks[f][i, j])]
        seg_maps = np.stack(seg_maps, axis=0)
        return seg_maps

    def _get_kpoint_heatmaps(self, idx, categories, sigma=4.0):
        """ Obtaining gripper location and generating a blob centered at it """
        num_hmaps = len(categories)
        hmaps = np.zeros((self.num_frames, num_hmaps, *self.hmap_size))
        ep_num = self._ep_num_from_id(self.image_ids[idx[0]])
        frame_nums = [self._frame_num_from_id(self.image_ids[id_]) for id_ in idx]
        object_poses = [self.object_poses[ep_num][frame_num] for frame_num in frame_nums]
        for fnum in range(self.num_frames):
            frame_obj_poses = object_poses[fnum]
            for i, cat in enumerate(categories):
                cat_kpoints = [self._to_img_plane(pos_3d) for pos_3d in frame_obj_poses[cat]]
                hmap_gen = HeatmapGenerator(shape=self.hmap_size, num_kpoints=len(cat_kpoints), sigma=sigma)
                cat_hmap = hmap_gen(cat_kpoints)
                cat_hmap = np.sum(cat_hmap, axis=0, keepdims=True)
                hmaps[fnum][i] = cat_hmap
        return hmaps if num_hmaps != 1 else hmaps[:, -1:]

    def _create_frame_to_obj_kpoints_dict(self):
        """ Linking data into a data structure """
        for scene_gt_fp in self.scene_gt_fps:
            ep_num = self._ep_num_from_fname(scene_gt_fp)
            with open(scene_gt_fp, 'r') as scene_json_file:
                ep_dict = json.load(scene_json_file)
            object_poses = []
            for frame_elem in ep_dict.values():
                frame_obj_dict = defaultdict(list)
                for obj_elem in frame_elem:
                    obj_id = obj_elem["obj_id"]
                    pos_3d = obj_elem["cam_t_m2c"]
                    frame_obj_dict[obj_id].append(pos_3d)
                object_poses.append(frame_obj_dict)
            self.object_poses[ep_num] = object_poses

    def _find_valid_sequences(self):
        """ Ensure that a sequence has the sufficient number of frames """
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        last_valid_idx = -1 * seq_len
        frame_offsets = range(0, self.num_frames * self.seq_step, self.seq_step)
        for idx in range(len(self.image_ids) - seq_len + 1):
            # handle overlapping sequences
            if (not self.allow_seq_overlap and (idx < last_valid_idx + seq_len)) \
                 or (self.allow_seq_overlap and (idx < last_valid_idx + seq_len//4)):
                continue

            ep_nums = [self._ep_num_from_id(self.image_ids[idx + offset]) for offset in frame_offsets]
            frame_nums = [self._frame_num_from_id(self.image_ids[idx + offset]) for offset in frame_offsets]
            # first few frames are discarded
            if frame_nums[0] < self.SKIP_FIRST_N:
                continue
            # last T frames of an episode should not be chosen as the start of a sequence
            if ep_nums[0] != ep_nums[-1]:
                continue

            gripper_obj_id = self.NUM_CLASSES
            gripper_pos = [self.object_poses[ep_nums[0]][frame_num][gripper_obj_id][0] for frame_num in frame_nums]

            # discard sequences where the gripper_pos is not down enough
            offset = self.GRIPPER_VALID_OFFSET
            gripper_xy = self._to_img_plane(gripper_pos[0])
            if not ((offset <= gripper_xy[0] <= 1.0 - offset) and (offset <= gripper_xy[1] <= 1.0 - offset)):
                continue
            gripper_xy = self._to_img_plane(gripper_pos[-1])
            if not ((offset <= gripper_xy[0] <= 1.0 - offset) and (offset <= gripper_xy[1] <= 1.0 - offset)):
                continue

            # discard sequences without considerable gripper movement
            gripper_pos_deltas = self._get_gripper_pos_xydist(gripper_pos)
            gripper_pos_deltas_above_min = [(delta > 1.0) for delta in gripper_pos_deltas]
            gripper_pos_deltas_below_max = [(delta < 30.0) for delta in gripper_pos_deltas]
            most = lambda lst, factor=0.67: sum(lst) >= factor * len(lst)
            gripper_movement_ok = most(gripper_pos_deltas_above_min) and all(gripper_pos_deltas_below_max)
            if not gripper_movement_ok:
                continue
            self.valid_idx.append(idx)
            last_valid_idx = idx
        assert len(self.valid_idx) > 0
        return

    def _to_img_plane(self, pos_3d):
        """ Pose3d to pose2d """
        Bin_H, Bin_W = self.BIN_SIZE
        px, py, _ = pos_3d
        x, y = px / Bin_W + 0.5, py / Bin_H + 0.5
        # discard objects dropped outside of the bin
        return (x, y) if ((0 <= x < 1.0) and (0 <= y < 1.0)) else (0., 0.)

    def _comp_gripper_pos(self, old, new):
        x_diff, y_diff = new[0] - old[0], new[1] - old[1]
        return math.sqrt(x_diff * x_diff + y_diff * y_diff)

    def _get_gripper_pos_xydist(self, gripper_pos):
        return [self._comp_gripper_pos(old, new) for old, new in zip(gripper_pos, gripper_pos[1:])]

    def _get_gripper_pos_diff(self, gripper_pos):
        gripper_pos_numpy = np.array(gripper_pos)
        return np.stack([new-old for old, new in zip(gripper_pos_numpy, gripper_pos_numpy[1:])], axis=0)

    def _ep_num_from_id(self, file_id: str):
        return int(file_id[-17:-11])

    def _frame_num_from_id(self, file_id: str):
        return int(file_id[-10:-4])

    def _ep_num_from_fname(self, file_name: str):
        return int(file_name[-20:-14])

#
