"""
Computation of different metrics
"""

import os
import json
import piqa
import lpips
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from lib.logger import print_
from lib.visualizations import visualize_metric
from CONFIG import METRICS, METRIC_SETS


class MetricTracker:
    """
    Class for computing several evaluation metrics
    """

    def __init__(self, metrics=["accuracy"]):
        """ Module initializer """
        assert isinstance(metrics, list), f"Metrics argument must be a list, and not {type(metrics)}"
        if isinstance(metrics, str):
            if metrics not in METRIC_SETS.keys():
                raise ValueError(f"If str, metrics must be one of {METRIC_SETS.keys()}, not {metrics}")
        else:
            for metric in metrics:
                if "pdj" in metric or "pck" in metric:
                    continue
                if metric not in METRICS:
                    raise NotImplementedError(f"Metric {metric} not implemented. Use one of {METRICS}")

        metrics = METRIC_SETS[metrics] if isinstance(metrics, str) else metrics
        self.metric_computers = {m: self._get_metric(m) for m in metrics}
        self.reset_results()
        return

    def reset_results(self):
        """ Reseting results and metric computers """
        self.results = {}
        for m in self.metric_computers.values():
            m.reset()
        return

    def accumulate(self, preds, targets):
        """ Computing the different metrics and adding them to the results list """
        for _, metric_computer in self.metric_computers.items():
            metric_computer.accumulate(preds=preds, targets=targets)
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        for metric, metric_computer in self.metric_computers.items():
            results = metric_computer.aggregate()
            if isinstance(results, dict):
                for key, val in results.items():
                    self.results[key] = val
            else:
                mean_metric, framewise_metric = results
                self.results[f"mean_{metric}"] = mean_metric
                self.results[f"framewise_{metric}"] = framewise_metric
        return

    def find_best_sample_results(self, num_last_results=1):
        """
        Finds the best metric among the last results, and possibly removes the rest.

        Args:
        -----
        num_last_results: int
            Number of last results to consider when searching for the best
        """
        if num_last_results >= 2:  # only considering best if there are more that 1 competing results
            for _, metric_computer in self.metric_computers.items():
                saved_values = metric_computer.values[:-num_last_results]
                competing_values = torch.stack(metric_computer.values[-num_last_results:])  # (Samps, B, F)
                if metric_computer.LOWER_BETTER:
                    best_val_idx = competing_values.sum(dim=-1).argmin(dim=0)
                else:
                    best_val_idx = competing_values.sum(dim=-1).argmax(dim=0)
                batch_ids = torch.arange(competing_values.shape[1])
                metric_computer.values = saved_values + [competing_values[best_val_idx, batch_ids]]
        return

    def summary(self, get_results=True):
        """ Printing and fetching the results """
        print_("RESULTS:")
        print_("--------")
        for metric in self.results.keys():
            if "mean" in metric:
                print_(f"  {metric}:  {round(self.results[metric].item(), 5)}")
        return self.results

    def save_results(self, results_file, summary=False):
        """ Storing results into JSON file """
        if summary:
            _ = self.summary()
        cur_results = {}
        for metric in self.results:
            if "framewise" in metric:
                cur_results[metric] = [round(r, 5) for r in self.results[metric].cpu().detach().tolist()]
            elif "mean" in metric:
                cur_results[metric] = round(self.results[metric].item(), 5)
            else:
                cur_results[metric] = self.results[metric].cpu().detach().tolist()

        with open(results_file, "w") as file:
            json.dump(cur_results, file)
        return

    def save_plots(self, savepath, frame_nums, start_frame=0, suffix=""):
        """ Saving metric plots """
        for metric in self.results:
            try:
                if "framewise" not in metric:
                    continue
                metric_name = metric.split("_")[-1] + suffix
                cur_vals = [round(r, 5) for r in self.results[metric].cpu().detach().tolist()]
                cur_savepath = os.path.join(savepath, f"results_{metric_name}.png")
                visualize_metric(
                        vals=cur_vals,
                        x_axis=frame_nums,
                        title=metric_name,
                        savepath=cur_savepath,
                        xlabel="Frame"
                    )
            except:
                print_(f"Error saving metric plots for metric {metric}...")

        return

    def _get_metric(self, metric):
        """ """
        # image-generation/video-prediction metrics
        if metric == "mse":
            metric_computer = MSE()
        elif metric == "mae":
            metric_computer = MAE()
        elif metric == "psnr":
            metric_computer = PSNR()
        elif metric == "ssim":
            metric_computer = SSIM()
        elif metric == "lpips":
            metric_computer = LPIPS()
        # keypoint and blob metrics
        elif metric == "mpjpe":
            metric_computer = MPJPE()
        elif "pdj" in metric:
            height_factor = float(metric.split("@")[-1])
            metric_computer = PDJ(body_height_factor=height_factor)
        elif "pck" in metric:
            height_factor = float(metric.split("@")[-1])
            metric_computer = PCK(body_height_factor=height_factor)
        # segmentation metrics
        elif metric == "segmentation_accuracy":
            metric_computer = SegmentationAccuracy()
        elif metric == "iou":
            metric_computer = IoU()
        else:
            raise NotImplementedError(f"Unknown metric {metric}. Use one of {METRICS} ...")
        return metric_computer


class Metric:
    """
    Base class for metrics
    """

    def __init__(self):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)

        # accounting for possible nan values
        nan_vals = all_values.isnan()
        all_values[nan_vals] = 0

        # safe mean
        mean_values = all_values.sum() / torch.logical_not(nan_vals).sum()
        frame_values = all_values.sum(dim=0) / torch.logical_not(nan_vals).sum(dim=0)
        return mean_values, frame_values

    def _shape_check(self, tensor, name="Preds"):
        """ """
        if len(tensor.shape) not in [3, 4, 5]:
            raise ValueError(f"{name} has shape {tensor.shape}, but it must have one of the folling shapes\n"
                             " - (B, F, C, H, W) for frame or heatmap prediction.\n"
                             " - (B, F, D) or (B, F, N_joints, N_coords) for pose skeleton prediction")


############################
# Video Prediction Metrics #
############################

class MSE(Metric):
    """ Mean Squared Error computer """

    LOWER_BETTER = True

    def __init__(self):
        """ """
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if len(preds.shape) == 5 and len(targets.shape) == 5:
            cur_mse = (preds - targets).pow(2).mean(dim=(-1, -2, -3))
        elif len(preds.shape) == 3 and len(targets.shape) == 3:
            cur_mse = (preds - targets).pow(2).mean(dim=-1)
        self.values.append(cur_mse)
        return


class MAE(Metric):
    """ Mean Absolute Error computer """

    LOWER_BETTER = True

    def __init__(self):
        """ """
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if len(preds.shape) == 5 and len(targets.shape) == 5:
            cur_mae = (preds - targets).abs().mean(dim=(-1, -2, -3))
        elif len(preds.shape) == 3 and len(targets.shape) == 3:
            cur_mae = (preds - targets).abs().mean(dim=-1)
        self.values.append(cur_mae)
        return


class PSNR(Metric):
    """ Peak Signal-to-Noise ratio computer """

    LOWER_BETTER = False

    def __init__(self):
        """ """
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_psnr = piqa.psnr.psnr(preds, targets)
        cur_psnr = cur_psnr.view(B, F)
        self.values.append(cur_psnr)
        return


class SSIM(Metric):
    """ Structural Similarity computer """

    LOWER_BETTER = False

    def __init__(self, window_size=11, sigma=1.5, n_channels=3):
        """ """
        self.ssim = piqa.ssim.SSIM(
                window_size=window_size,
                sigma=sigma,
                n_channels=n_channels,
                reduction=None
            )
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if self.ssim.kernel.device != preds.device:
            self.ssim = self.ssim.to(preds.device)

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_ssim = self.ssim(preds, targets)
        cur_ssim = cur_ssim.view(B, F)
        self.values.append(cur_ssim)
        return


class LPIPS(Metric):
    """ Learned Perceptual Image Patch Similarity computers """

    LOWER_BETTER = True

    def __init__(self, network="alex", pretrained=True, reduction=None):
        """ """
        self.lpips = lpips.LPIPS(net='alex')
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if not hasattr(self.lpips, "device") or self.lpips.device != preds.device:
            self.lpips = self.lpips.to(preds.device)
            self.lpips.device = preds.device

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_lpips = self.lpips(preds, targets)
        cur_lpips = cur_lpips.view(B, F)
        self.values.append(cur_lpips)
        return


############################
# POSE FORECASTING METRICS #
############################

class MPJPE(Metric):
    """ 'Mean Per-Joint Position Error' metric for keypoint prediction """

    MIN_CONF = 0.03
    LOWER_BETTER = True

    def __init__(self):
        """ """
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        B, F, N_kpts, H, W = preds.shape

        # extracting X and Y of heatmaps
        preds_val, preds_idx = torch.max(preds.view(B * F * N_kpts, -1), dim=-1)
        targets_val, targets_idx = torch.max(targets.view(B * F * N_kpts, -1), dim=-1)
        preds_y, preds_x = torch.div(preds_idx, W, rounding_mode="floor"), preds_idx % W
        targets_y, targets_x = torch.div(targets_idx, W, rounding_mode="floor"), targets_idx % W

        # computing position error
        dist = ((preds_y - targets_y).pow(2) + (preds_x - targets_x).pow(2)).sqrt()
        dist = dist.view(B, F, N_kpts)

        # finding number of valid keypoints and averaging
        preds_valid = preds_val >= self.MIN_CONF
        targets_valid = targets_val > 1e-6
        total_valid = torch.logical_and(preds_valid, targets_valid).view(B, F, N_kpts)

        dist[torch.logical_not(total_valid)] = 0
        mean_dist = dist.sum(dim=-1) / total_valid.sum(dim=-1)
        self.values.append(mean_dist)
        return


class PDJ(Metric):
    """ 'Percentage of Detected Joints' metric for keypoint prediction """

    MIN_CONF = 0.03
    LOWER_BETTER = False

    def __init__(self, body_height_factor=0.1):
        """ """
        self.height_factor = body_height_factor
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        B, F, N_kpts, H, W = preds.shape

        # extracting X and Y of heatmaps
        preds_val, preds_idx = torch.max(preds.view(B * F, N_kpts, -1), dim=-1)
        targets_val, targets_idx = torch.max(targets.view(B * F, N_kpts, -1), dim=-1)
        preds_y, preds_x = torch.div(preds_idx, W, rounding_mode="floor"), preds_idx % W
        targets_y, targets_x = torch.div(targets_idx, W, rounding_mode="floor"), targets_idx % W

        batch_values = []
        for b in range(B * F):
            # computing body heaight for current frame
            min_y, max_y = H, 0
            for k in range(N_kpts):
                y = targets_y[b, k]
                if targets_val[b, k] > 0:
                    max_y, min_y = max(max_y, y), min(min_y, y)
            body_height = abs(max_y - min_y)
            min_dist = self.height_factor * body_height if body_height > 0 else self.height_factor

            # finding number of valid keypoint
            preds_valid = preds_val[b] >= self.MIN_CONF
            targets_valid = targets_val[b] > 1e-6
            total_valid = torch.logical_and(preds_valid, targets_valid)

            # computing number of keypoints within threshold
            dist = ((preds_y[b] - targets_y[b]).pow(2) + (preds_x[b] - targets_x[b]).pow(2)).sqrt()
            num_correct_preds = (dist[total_valid] <= min_dist).sum()
            pdj = num_correct_preds / (targets_valid).sum()
            batch_values.append(pdj)

        pdj = torch.stack(batch_values).view(B, F)
        self.values.append(pdj)
        return


class PCK(Metric):
    """ 'Percentage of Correct Keypoints' metric for keypoint prediction """

    MIN_CONF = 0.03
    LOWER_BETTER = False

    def __init__(self, body_height_factor=0.1):
        """ """
        self.height_factor = body_height_factor
        super().__init__()

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        B, F, N_kpts, H, W = preds.shape

        # extracting X and Y of heatmaps
        preds_val, preds_idx = torch.max(preds.view(B * F, N_kpts, -1), dim=-1)
        targets_val, targets_idx = torch.max(targets.view(B * F, N_kpts, -1), dim=-1)
        preds_y, preds_x = torch.div(preds_idx, W, rounding_mode="floor"), preds_idx % W
        targets_y, targets_x = torch.div(targets_idx, W, rounding_mode="floor"), targets_idx % W

        batch_values = []
        for b in range(B * F):
            # computing body heaight for current frame
            min_y, max_y = H, 0
            for k in range(N_kpts):
                y = targets_y[b, k]
                if targets_val[b, k] > 0:
                    max_y, min_y = max(max_y, y), min(min_y, y)
            body_height = abs(max_y - min_y)
            min_dist = self.height_factor * body_height if body_height > 0 else self.height_factor

            # finding number of valid keypoint
            preds_valid = preds_val[b] >= self.MIN_CONF
            targets_valid = targets_val[b] > 1e-6
            total_valid = torch.logical_and(preds_valid, targets_valid)

            # computing number of keypoints within threshold
            dist = ((preds_y[b] - targets_y[b]).pow(2) + (preds_x[b] - targets_x[b]).pow(2)).sqrt()
            num_correct_preds = (dist[total_valid] <= min_dist).sum()
            pdj = num_correct_preds / (preds_valid).sum()
            batch_values.append(pdj)

        pdj = torch.stack(batch_values).view(B, F)
        self.values.append(pdj)
        return


############################
#   SEGMENTATION METRICS   #
############################

class SegmentationAccuracy(Metric):
    """ Computing average and per-class segmentation accuracy"""

    LOWER_BETTER = False

    def __init__(self):
        """ """
        self.classwise_values = []
        self.values = []
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.values = []
        self.classwise_values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        B, F, N_classes, H, W = preds.shape
        preds, targets = preds.view(B * F, N_classes, -1), targets.view(B * F, N_classes, -1)
        pred_classes, target_classes = torch.argmax(preds, dim=1), torch.argmax(targets, dim=1)

        accs, per_cls_accs = [], []
        for i in range(B * F):
            cm = compute_confusion_matrix(
                    targets=pred_classes[i],
                    preds=target_classes[i],
                    num_classes=N_classes
                )
            per_cls_correct_preds = cm.diag()
            per_cls_targets = cm.sum(dim=-1)

            per_cls_acc = per_cls_correct_preds / per_cls_targets
            acc = per_cls_correct_preds.sum() / per_cls_targets.sum()
            accs.append(acc.item())
            per_cls_accs.append(per_cls_acc)
        self.values.append(torch.tensor(accs).view(B, F))
        self.classwise_values.append(torch.stack(per_cls_accs).view(B, F, N_classes))
        return

    def aggregate(self):
        """ Computing average metric, bpth global and framewise, and overall and per-class"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)

        # accounting for possible nan values
        all_cls_values = torch.cat(self.classwise_values, dim=0)
        nan_vals = all_cls_values.isnan()
        all_cls_values[nan_vals] = 0
        mean_classwise_values = all_cls_values.sum(dim=(0, 1)) / torch.logical_not(nan_vals).sum(dim=(0, 1))
        nan_vals = mean_classwise_values.isnan()
        mean_classwise_values[nan_vals] = 0

        results = {
            "mean_segmentation_accuracy": mean_values,
            "framewise_segmentation_accuracy": frame_values,
            "classwise_segmentation_accuracy": mean_classwise_values,
        }
        return results


class IoU(Metric):
    """ Computing average and per-class intersection-over-union"""

    LOWER_BETTER = False

    def __init__(self):
        """ """
        self.classwise_values = []
        self.values = []
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.values = []
        self.classwise_values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        B, F, N_classes, H, W = preds.shape
        preds, targets = preds.view(B * F, N_classes, -1), targets.view(B * F, N_classes, -1)
        pred_classes, target_classes = torch.argmax(preds, dim=1), torch.argmax(targets, dim=1)

        ious, per_cls_ious = [], []
        for i in range(B * F):
            cm = compute_confusion_matrix(
                    targets=pred_classes[i],
                    preds=target_classes[i],
                    num_classes=N_classes
                )
            per_cls_correct_preds = cm.diag()
            per_cls_targets = cm.sum(dim=-1)
            per_cls_preds = cm.sum(dim=0)

            union_preds_targets = per_cls_targets + per_cls_preds - per_cls_correct_preds
            class_iou = per_cls_correct_preds / union_preds_targets
            ious.append(class_iou.mean().item())
            per_cls_ious.append(class_iou)

        self.values.append(torch.tensor(ious).view(B, F))
        self.classwise_values.append(torch.stack(per_cls_ious).view(B, F, N_classes))
        return

    def aggregate(self):
        """ Computing average metric, both global and framewise, and overall and per-class"""
        all_values = torch.cat(self.values, dim=0)
        nan_vals = all_values.isnan()
        all_values[nan_vals] = 0
        mean_values = all_values.sum() / torch.logical_not(nan_vals).sum()
        frame_values = all_values.sum(dim=0) / torch.logical_not(nan_vals).sum(dim=0)

        # accounting for possible nan values
        all_cls_values = torch.cat(self.classwise_values, dim=0)
        nan_vals = all_cls_values.isnan()
        all_cls_values[nan_vals] = 0
        mean_cls_values = all_cls_values.sum(dim=(0, 1)) / torch.logical_not(nan_vals).sum(dim=(0, 1))
        nan_vals = mean_cls_values.isnan()
        mean_cls_values[nan_vals] = 0

        results = {
            "mean_iou": mean_values,
            "framewise_iou": frame_values,
            "classwise_iou": mean_cls_values
        }
        return results


def compute_confusion_matrix(preds, targets, num_classes=3):
    """ Computing confusion matrix """
    if not torch.is_tensor(preds) or not torch.is_tensor(targets):
        preds, targets = torch.tensor(preds).flatten(), torch.tensor(targets).flatten()
    preds, targets = preds.cpu(), targets.cpu()
    cm = confusion_matrix(targets, preds, labels=np.arange(num_classes))
    cm = torch.from_numpy(cm)
    return cm


#
