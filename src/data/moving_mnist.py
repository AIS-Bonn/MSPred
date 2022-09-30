"""
Implementation of moving-mnist and custom-moving-mnist datasets
"""

import os
import numpy as np
import torch
from torchvision.datasets import MNIST
from CONFIG import CONFIG, METRIC_SETS
from .base_dataset import SequenceDataset
from data.heatmaps import HeatmapGenerator


class MovingMNIST(SequenceDataset):
    """
    Precomputed Moving MNIST dataset loader.
    """

    MAX_SEQ_LEN = 60
    NUM_TRAIN_SEQ = 10000
    NUM_TEST_SEQ = 10000
    NUM_HMAP_CHANNELS = [1, 1]
    STRUCT_TYPE = "BLOB_HEATMAP"

    METRICS_LEVEL_0 = METRIC_SETS["video_prediction"]
    METRICS_LEVEL_1 = METRIC_SETS["blob"]
    METRICS_LEVEL_2 = METRIC_SETS["blob"]

    train_to_val_ratio = 0.99

    def __init__(self, split="train", num_frames=16, num_channels=3, img_size=64):
        """ Dataset initializer """
        assert num_frames <= self.MAX_SEQ_LEN
        assert num_channels in [1, 3]
        assert split in ["train", "val", "test"]
        data_path = CONFIG["paths"]["data_path"]
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.data_dir = os.path.join(data_path, "moving_mnist", ("test" if split == "test" else "train"))
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"MovingMNIST path {self.data_dir} does not exists...")

        num_train_seq = int(self.NUM_TRAIN_SEQ * self.train_to_val_ratio)
        seq_range = {}
        seq_range["train"] = range(0, num_train_seq)
        seq_range["val"] = range(num_train_seq, self.NUM_TRAIN_SEQ)
        seq_range["test"] = range(0, self.NUM_TEST_SEQ)
        self.seq_range = seq_range[split]
        return

    def __getitem__(self, i):
        """ Sampling sequence from dataset """
        seq_num = self.seq_range[i]
        seq_num = self._idx_to_str(seq_num)
        npy_files = [f"seq_{seq_num}.npy", f"seq_pos_{seq_num}.npy", f"seq_hm_{seq_num}.npy"]
        # load frames
        seq_path = os.path.join(self.data_dir, npy_files[0])
        frames = np.load(seq_path)
        frames = frames[:self.num_frames] / 255.
        frames = np.expand_dims(frames, axis=1)
        if self.num_channels == 3:
            frames = np.repeat(frames, 3, axis=1)
        # load center-point heatmaps
        heatmaps = []
        for n in range(2):
            hm_path = os.path.join(self.data_dir, npy_files[n+1])
            hmap = np.load(hm_path)
            hmap = hmap[:self.num_frames]
            hmap = np.expand_dims(hmap, axis=1)
            heatmaps.append(hmap)
        frames = torch.Tensor(frames)
        heatmaps = [torch.Tensor(hmap) for hmap in heatmaps]
        return {"frames": frames, "heatmaps": heatmaps}

    def __len__(self):
        """ """
        return len(self.seq_range)

    def _idx_to_str(self, idx):
        """ """
        idx_str = str(idx)
        idx_str = idx_str.zfill(5)
        return idx_str


class CustomMovingMNIST(SequenceDataset):
    """
    Custom MovingMNIST dataset that generates data on-the-fly.
    """

    NUM_HMAP_CHANNELS = [1, 1]
    STRUCT_TYPE = "BLOB_HEATMAP"

    MOVING_SPECS = {
        "speed_min": 2,
        "speed_max": 5,
        "acc_min": 0,
        "acc_max": 0
    }

    METRICS_LEVEL_0 = METRIC_SETS["video_prediction"]
    METRICS_LEVEL_1 = METRIC_SETS["blob"]
    METRICS_LEVEL_2 = METRIC_SETS["blob"]

    def __init__(self, split="train", num_frames=30, num_channels=3, num_digits=2, img_size=64):
        """ Initializer of the moving MNIST dataset """
        # arguments
        assert split in ['train', 'val', 'test']
        assert num_channels in [1, 3]
        self.n_frames = num_frames
        self.split = split
        self.num_channels = num_channels
        self.num_digits = num_digits
        self.img_size = img_size

        # loading data
        data_path = CONFIG["paths"]["data_path"]
        train = (self.split == "train")
        self.data = MNIST(root=data_path, train=train, download=True)

        # loading moving parameters
        speed_max, acc_max = self.MOVING_SPECS["speed_max"], self.MOVING_SPECS["acc_max"]
        if split == "train" or split == "val":
            self.get_speed = lambda: np.random.randint(-1*speed_max, speed_max+1)
            self.get_acc = lambda: np.random.randint(-1*acc_max, acc_max+1)
            self.get_init_pos = lambda img_size, digit_size: np.random.randint(0, img_size-digit_size)
        elif split == "test":
            rng_speed = np.random.default_rng(12345)
            rng_acc = np.random.default_rng(12345)
            rng_pos = np.random.default_rng(12345)
            self.rng_digit = np.random.default_rng(12345)
            self.get_speed = lambda: rng_speed.integers(-1*speed_max, speed_max+1)
            self.get_acc = lambda: rng_acc.integers(-1*acc_max, acc_max+1)
            self.get_init_pos = lambda img_size, digit_size: rng_pos.integers(0, img_size-digit_size)

        # 0: digit center point heatmaps, 1: digit center blob heatmaps
        self.generate_hms = [HeatmapGenerator((img_size, img_size), num_digits, sigma=sigma) for sigma in [2.0, 5.0]]
        return

    def __getitem__(self, idx):
        """ Sampling sequence """
        frames = np.zeros((self.n_frames, self.img_size, self.img_size, self.num_channels))

        digits, next_poses, speeds = [], [], []
        for i in range(self.num_digits):
            digit, pos, speed = self._sample_digit(idx+i)
            digits.append(digit)
            next_poses.append(pos)
            speeds.append(speed)

        # generating sequence by moving the digit given velocity
        positions = []
        for i, frame in enumerate(frames):
            for j, (digit, cur_pos, speed) in enumerate(zip(digits, next_poses, speeds)):
                digit_size = digit.shape[-2]
                speed, cur_pos = self._move_digit(
                        speed=speed,
                        cur_pos=cur_pos,
                        img_size=self.img_size,
                        digit_size=digit_size
                    )
                speeds[j] = speed
                next_poses[j] = cur_pos
                frame[cur_pos[0]:cur_pos[0]+digit_size, cur_pos[1]:cur_pos[1]+digit_size] += digit
            frames[i] = np.clip(frame, 0, 1)
            positions.append([(np.flip(p) + digit_size/2) / self.img_size for p in next_poses])
        frames = torch.Tensor(frames).permute(0, 3, 1, 2)

        hmaps = [np.zeros((self.n_frames, self.num_digits, self.img_size, self.img_size)) for i in range(2)]
        for i, pos in enumerate(positions):
            for h in range(2):
                hmaps[h][i] = self.generate_hms[h](pos)
        hmaps = [torch.Tensor(np.sum(hmap, axis=1, keepdims=True)) for hmap in hmaps]
        return {"frames": frames, "heatmaps": hmaps}

    def _sample_digit(self, idx):
        """ Sampling digit, original position and speed """
        if self.split == "test":
            digit_id = self.rng_digit.integers(len(self.data))
        else:
            digit_id = np.random.randint(len(self.data))
        cur_digit = np.array(self.data[digit_id][0]) / 255  # sample IDX, digit
        digit_size = cur_digit.shape[-1]
        cur_digit = cur_digit[..., np.newaxis]
        if self.num_channels == 3:
            cur_digit = np.repeat(cur_digit, 3, axis=-1)

        # obtaining position in original frame
        x_coord = self.get_init_pos(self.img_size, digit_size)
        y_coord = self.get_init_pos(self.img_size, digit_size)
        cur_pos = np.array([y_coord, x_coord])

        # generating sequence
        speed_x, speed_y, acc = None, None, None
        while speed_x is None or np.abs(speed_x) < self.MOVING_SPECS["speed_min"]:
            speed_x = self.get_speed()
        while speed_y is None or np.abs(speed_y) < self.MOVING_SPECS["speed_min"]:
            speed_y = self.get_speed()
        while acc is None or np.abs(acc) < self.MOVING_SPECS["acc_min"]:
            acc = self.get_acc()
        speed = np.array([speed_y, speed_x])

        return cur_digit, cur_pos, speed

    def _move_digit(self, speed, cur_pos, img_size, digit_size):
        """ Performing digit movement. Also producing bounce and making appropriate changes """
        next_pos = cur_pos + speed
        for i, p in enumerate(next_pos):
            # left/down bounce
            if (p + digit_size > img_size):
                offset = p + digit_size - img_size
                next_pos[i] = p - offset
                speed[i] = -1 * speed[i]
            # right/up bounce
            elif (p < 0):
                next_pos[i] = -1 * p
                speed[i] = -1 * speed[i]

        return speed, next_pos

    def __len__(self):
        """ """
        return len(self.data)

#
