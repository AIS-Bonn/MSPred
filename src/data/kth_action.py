"""
KTH-Actions dataset, including frames, pose keypoints and locations
"""

import random
import os
import json
import numpy as np
import torch
import imageio
import torchfile
from CONFIG import CONFIG, METRIC_SETS
from .base_dataset import SequenceDataset
from data.heatmaps import HeatmapGenerator


# Helper functions
def _read_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def _swap(L, i1, i2):
    L[i1], L[i2] = L[i2], L[i1]


class KTH(SequenceDataset):
    """
    KTH-Actions dataset. We obtain a sequence of frames with the corresponding body-joints in
    heatmap form, as well as the location of the person as a blob
    """
    KPOINTS = [0, 2, 5, 4, 7, 9, 12, 10, 13, 1]
    NUM_KPOINTS = len(KPOINTS)
    KPT_TO_IDX = {0: 0, 2: 1, 5: 2, 4: 3, 7: 4, 9: 5, 12: 6, 10: 7, 13: 8, 1: 9}
    SWAP_PAIRS = [(2, 5), (4, 7), (9, 12), (10, 13)]
    HARD_KPTS_PER_CLASS = {
            "boxing": [4, 7],
            "handclapping": [4, 7],
            "handwaving": [4, 7],
            "walking": [9, 12, 10, 13],
            "running": [9, 12, 10, 13],
            "jogging": [9, 12, 10, 13]
    }
    CLASSES = list(HARD_KPTS_PER_CLASS.keys())

    # classes with relatively shorter sequences
    SHORT_CLASSES = ['walking', 'running', 'jogging']
    MIN_SEQ_LEN = 29  # 14, 29, 49

    NUM_HMAP_CHANNELS = [NUM_KPOINTS - 1, 1]
    STRUCT_TYPE = "KEYPOINT_BLOBS"

    ALL_IDX = None
    IDX_TO_CLS_VID_SEQ = None
    train_to_val_ratio = 0.98
    first_frame_rng_seed = 1234

    METRICS_LEVEL_0 = METRIC_SETS["video_prediction"]
    METRICS_LEVEL_1 = METRIC_SETS["keypoint"]
    METRICS_LEVEL_2 = METRIC_SETS["single_keypoint_metric"]

    def __init__(self, split, num_frames=50, num_channels=3, img_size=64, horiz_flip_aug=True):
        """ Dataset initializer"""
        assert split in ['train', 'val', 'test']
        data_path = CONFIG["paths"]["data_path"]
        self.data_root = os.path.join(data_path, f"KTH_{img_size}/processed")
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"KTH-Data does not exist in {self.data_root}...")

        self.split = split
        self.n_frames = num_frames
        self.num_channels = num_channels
        self.img_size = img_size
        self.horiz_flip_aug = horiz_flip_aug and self.split != "test"

        dataset = "train" if self.split != "test" else "test"
        self.dataset = dataset
        self.data = {}
        self.keypoints = {}
        for c in self.CLASSES:
            data_fname = os.path.join(self.data_root, c, f'{dataset}_meta{img_size}x{img_size}.t7')
            kpt_fname = os.path.join(self.data_root, c, f'{dataset}_keypoints{img_size}x{img_size}.json')
            self.data[c] = torchfile.load(data_fname)
            self.keypoints[c] = _read_json(kpt_fname)
        # 0: pose-keypoints' heatmap; 1: upper-torso keypoint heatmap
        self.generate_hmaps = [HeatmapGenerator((img_size, img_size), self.NUM_HMAP_CHANNELS[i], 1) for i in range(2)]

        # list of valid (cls, vid_idx, seq_idx) tuples
        if KTH.ALL_IDX is None:
            KTH.IDX_TO_CLS_VID_SEQ = self._find_valid_sequences()
            KTH.ALL_IDX = list(range(0, len(KTH.IDX_TO_CLS_VID_SEQ)))
        if self.split == "train":
            random.shuffle(KTH.ALL_IDX)
        self.idx_list = KTH.ALL_IDX
        if self.split != "test":
            train_len = int(len(KTH.ALL_IDX) * self.train_to_val_ratio)
            self.idx_list = self.idx_list[:train_len] if self.split == "train" else self.idx_list[train_len:]

    def _is_valid_sequence(self, seq, cls):
        """ Exploit short sequences of specific classes by extending them with repeated last frame """
        extend_seq = (cls in self.SHORT_CLASSES and len(seq) >= self.MIN_SEQ_LEN)
        return (len(seq) >= self.n_frames or extend_seq)

    def _find_valid_sequences(self):
        """ Ensure that a sequence has the sufficient number of frames """
        idx_to_cls_vid_seq = []
        for cls, cls_data in self.data.items():
            for vid_idx, vid in enumerate(cls_data):
                vid_seq = vid[b'files']
                for seq_idx, seq in enumerate(vid_seq):
                    if self._is_valid_sequence(seq, cls):
                        idx_to_cls_vid_seq.append((cls, vid_idx, seq_idx))
        return idx_to_cls_vid_seq

    def __getitem__(self, i):
        """ Sampling sequence from the dataset """
        i = self.idx_list[i]
        cls, vid_idx, seq_idx = KTH.IDX_TO_CLS_VID_SEQ[i]
        vid = self.data[cls][vid_idx]
        seq = vid[b'files'][seq_idx]

        # initializing arrays for images, kpts, and blobs
        cls_kps = self.keypoints[cls]
        dname = os.path.join(self.data_root, cls, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.n_frames, self.img_size, self.img_size, self.num_channels))
        hmaps = [
                np.zeros((self.n_frames, self.NUM_HMAP_CHANNELS[0], self.img_size, self.img_size)),
                np.zeros((self.n_frames, self.NUM_HMAP_CHANNELS[1], self.img_size, self.img_size))
            ]

        # getting random starting idx, and corresponding data
        first_frame = 0
        if len(seq) > self.n_frames:
            rand_gen = random.Random(self.first_frame_rng_seed) if self.split == "test" else random
            first_frame = rand_gen.randint(0, len(seq) - self.n_frames)
        last_frame = (len(seq) - 1) if (len(seq) <= self.n_frames) else (first_frame + self.n_frames - 1)
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            im = imageio.imread(fname) / 255.
            if self.num_channels == 1:
                im = im[:, :, 0][:, :, np.newaxis]
            frames[i - first_frame] = im
            full_fname = os.path.join(vid[b'vid'].decode('utf-8'), seq[i].decode('utf-8'))
            frame_kpts = cls_kps[full_fname]
            for h, kpts in enumerate([frame_kpts[:-1], frame_kpts[-1:]]):
                hmaps[h][i-first_frame] = self.generate_hmaps[h](kpts)

        for i in range(last_frame + 1, self.n_frames):
            frames[i] = frames[last_frame]
            for h in range(2):
                hmaps[h][i] = hmaps[h][last_frame]

        frames = torch.Tensor(frames).permute(0, 3, 1, 2)
        hmaps = [torch.Tensor(hmap) for hmap in hmaps]
        # random horizontal flip augmentation
        if self.horiz_flip_aug and (random.randint(0, 1) == 0):
            frames, hmaps = self._horiz_flip(frames, hmaps)
        return {'frames': frames, 'heatmaps': hmaps, 'classes': cls}

    def _horiz_flip(self, frames, hmaps):
        """ Horizontal flip augmentation """
        frames = torch.flip(frames, dims=[3])
        assert len(hmaps) == 2
        hmaps_1, hmaps_2 = hmaps
        hmaps_1 = torch.flip(hmaps_1, dims=[3])
        hmaps_2 = torch.flip(hmaps_2, dims=[3])

        # swap symmetric keypoint channels
        kpoint_order = list(range(self.NUM_HMAP_CHANNELS[0]))
        for (k1, k2) in self.SWAP_PAIRS:
            i1 = self.KPT_TO_IDX[k1]
            i2 = self.KPT_TO_IDX[k2]
            _swap(kpoint_order, i1, i2)
        hmaps_1 = hmaps_1[:, kpoint_order]
        return frames, (hmaps_1, hmaps_2)

    def __len__(self):
        """ """
        return len(self.idx_list)

    def get_heatmap_weights(self, w_easy_kpts=1.0, w_hard_kpts=1.0):
        """ Getting specific weights for different keypoints """
        weights = {}
        for cls in self.CLASSES:
            weights[cls] = [w_easy_kpts] * self.NUM_HMAP_CHANNELS[0]
            hard_kpts = self.HARD_KPTS_PER_CLASS[cls]
            for kpt in hard_kpts:
                i = self.KPT_TO_IDX[kpt]
                weights[cls][i] = w_hard_kpts
        return weights

#
