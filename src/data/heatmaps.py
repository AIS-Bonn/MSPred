"""
Modules for generating Gaussian blobs/heatmaps given the center of the heatmap
and a variance. Used to generate pose-joint heatmap and location blobs
"""
import numpy as np


class SquareHeatmapGenerator():
    """
    Generates heatmaps on a square canvas.
        Source: https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/data/MPII/dp.py

    Args:
    -----
    size: int
        size of the canvas where heatmap is added
    num_kpoints: int
        Number of keypoints/heatmaps to generate
    sigma: float
        variance of the heatmaps
    """

    def __init__(self, size, num_kpoints, sigma=1.0):
        """ Generator initializer """
        self.size = size
        self.num_kpoints = num_kpoints
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.gaussian = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        """
        Generating heatmaps given keypoints

        Args:
        -----
        keypoints: list
            list containing N tuples with (x, y) coordinates of joints. [(x1, y1), ..., (xN, yN)]
        """
        assert len(keypoints) == self.num_kpoints
        num_channels = max(1, self.num_kpoints)
        hmaps = np.zeros(shape=(num_channels, self.size, self.size), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            x, y = int(pt[0] * self.size), int(pt[1] * self.size)
            assert (0 <= x < self.size) and (0 <= y < self.size), f"Wrong keypoint ({x}, {y})..."
            if x == 0 or y == 0:
                continue
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], self.size) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.size) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.size)
            aa, bb = max(0, ul[1]), min(br[1], self.size)
            hmaps[idx, aa:bb, cc:dd] = np.maximum(hmaps[idx, aa:bb, cc:dd], self.gaussian[a:b, c:d])
        return hmaps


class HeatmapGenerator():
    """
    Uses SquareHeatmapGenerator to create non-square heatmaps.

    Args:
    -----
    size: int
        size of the canvas where heatmap is added
    num_kpoints: int
        Number of keypoints/heatmaps to generate
    sigma: float
        variance of the heatmaps
    """

    def __init__(self, shape, num_kpoints, sigma=1.0):
        """ Generator initializer """
        self.shape = shape
        H, W = shape
        assert H <= W
        self.num_kpoints = num_kpoints
        self.generator = SquareHeatmapGenerator(H, num_kpoints, sigma)

    def __call__(self, keypoints):
        """
        Generating heatmaps centered at the keypoint coordinates.
          1. Generating heatmaps on a square array using SquareHeatmapGenerator
          2. Adding the square heatmap array to the total array shape

        Args:
        -----
        keypoints: list
            list containing N tuples with (x, y) coordinates of joints. [(x1, y1), ..., (xN, yN)]
        """
        if keypoints == []:
            keypoints = [(0., 0.)] * self.num_kpoints
        assert len(keypoints) == self.num_kpoints
        H, W = self.shape
        if H == W:
            sq_hmaps = self.generator(keypoints)
            return sq_hmaps
        x_ranges = []
        zoomed_kpoints = []
        for i, kpt in enumerate(keypoints):
            x = int(kpt[0] * W)
            if x == 0:
                x_ranges.append(None)
                zoomed_kpoints.append((0., 0.))
                continue
            if x <= H//2:
                r0 = 0
                kx = x
            elif x >= W - H//2:
                r0 = W - H
                kx = x - W + H
            else:
                r0 = x - H//2
                kx = H//2
            x_ranges.append(range(r0, r0 + H))
            zoomed_kpoints.append((kx/H, kpt[1]))
        sq_hmaps = self.generator(zoomed_kpoints)
        hmaps = np.zeros(shape=(self.num_kpoints, H, W), dtype=np.float32)
        for i in range(len(keypoints)):
            if x_ranges[i] is not None:
                hmaps[i][:, x_ranges[i]] = sq_hmaps[i]
        return hmaps

#
