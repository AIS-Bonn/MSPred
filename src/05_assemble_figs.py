"""
Assembling nice figures and GIFs from pre-saved ground-truth and predicted images
"""

import os
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

from lib.arguments import get_fig_generator_arguments
from lib.config import Config
from lib.logger import Logger, print_
import lib.utils as utils
import lib.visualizations as visualizations


NAMES = {
    "kth": ["Human Poses", "Body Position"],
    "synpick": ["Segmentation", "Gripper Position"]
}
CONTEXT = {
    "kth": [0, 4, 8],
    "synpick": [0, 2, 3]
}
SIZE = {
    "kth": (10, 8),
    "synpick": (12, 6)
}
SUBSIZE = {
    "kth": {
            "left": 0.0,
            "bottom": 0.15,
            "right": 1.0,
            "top": 1.0,
            "wspace": 0.01,
            "hspace": 0.1
        },
    "synpick": {
            "left": 0.0,
            "bottom": 0.15,
            "right": 1.0,
            "top": 1.0,
            "wspace": 0.01,
            "hspace": 0.1
        }
}
FRAME_OFF = {
    "kth": -0.16,
    "synpick": 0.3
}
Y_OFF = {
    "kth": 0.4,
    "synpick": 0.4
}
X_OFF = {
    "kth": 0.4,
    "synpick": 0.4
}

font = {
        'family': 'Times New Roman',
        'size': 20
    }
matplotlib.rc('font', **font)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def main(exp_path, checkpoint):
    """ Assembling figures and GIFs given precomputed images """
    cfg = Config(exp_path)
    exp_params = cfg.load_exp_config_file()
    dataset_name = exp_params["dataset"]["dataset_name"]
    context = exp_params["eval"]["context"]
    periods = exp_params["model"]["HierarchLSTM"]["periods"]

    # paths
    results_dir = os.path.join(exp_path, "results")
    plots_path = os.path.join(results_dir, f"plots_{checkpoint}")
    gif_resources_path = os.path.join(plots_path, "gif_resources")
    imgs_path = os.path.join(plots_path, "images")
    gifs_path = os.path.join(plots_path, "gifs")
    if not os.path.exists(gif_resources_path):
        raise FileNotFoundError(f"Resources to build figs. do not exists in {gif_resources_path}...")
    if len(os.listdir(gif_resources_path)) == 0:
        raise ValueError(f"Resources path {gif_resources_path} is empty...")

    num_imgs = len(os.listdir(gif_resources_path)) // 2
    for i in tqdm(range(num_imgs)):
        preds = [os.path.join(gif_resources_path, f"seq_{i+1}", f"hier_{j}") for j in range(3)]
        gts = [os.path.join(gif_resources_path, f"seq_{i+1}_gt", f"hier_{j}") for j in range(3)]
        # loading data
        seq_hier = {}
        for level, pred, gt in zip(range(3), preds, gts):
            seq_pred = utils.read_seq(pred)
            seq_gt = utils.read_seq(gt)
            seq_hier[level] = (seq_gt, seq_pred)

        # assembling gif
        savepath = os.path.join(gifs_path, f"combined_gif_{i+1}.gif")
        visualizations.make_gif_hierarch(
                gif_frames=seq_hier,
                context=context,
                savepath=savepath,
                gif_names=["Ground-Truth", "Predicted"],
                periods=periods,
                pad=2,
                interval=100
            )

        # assembling figure
        fig, ax = plt.subplots(nrows=6, ncols=8)
        fig.set_size_inches(w=SIZE[dataset_name][0], h=SIZE[dataset_name][1])
        for j in range(5):
            ax[0, 3+j].imshow(seq_hier[2][1][-5+j])
            ax[1, 3+j].imshow(seq_hier[2][0][-5+j])
        for j in range(5):
            ax[2, 3+j].imshow(seq_hier[1][1][-5+j])
            ax[3, 3+j].imshow(seq_hier[1][0][-5+j])
        for j in range(5):
            ax[4, 3+j].imshow(seq_hier[0][1][-5+j])

        disp = [seq_hier[0][0][k] for k in CONTEXT[dataset_name]] + seq_hier[0][0][-5:]
        for j in range(8):
            ax[5, j].imshow(disp[j])

        for rnum, row in enumerate(ax):
            for cnum, f in enumerate(row):
                f.axis("off")
                f.set_aspect("auto")

        y_off = Y_OFF[dataset_name]
        x_off = X_OFF[dataset_name]
        ax[0, 1].text(x=x_off, y=y_off/2, s=f"Predicted\n{NAMES[dataset_name][1]}")
        ax[1, 1].text(x=x_off, y=y_off/2, s=f"Ground-Truth\n{NAMES[dataset_name][1]}")
        ax[2, 1].text(x=x_off, y=y_off/2, s=f"Predicted\n{NAMES[dataset_name][0]}")
        ax[3, 1].text(x=x_off, y=y_off/2, s=f"Ground-Truth\n{NAMES[dataset_name][0]}")
        ax[4, 1].text(x=FRAME_OFF[dataset_name], y=y_off, s="Predicted Frames")

        ax[5, 1].text(x=-x_off, y=-2 * y_off, s="Context Frames", transform=ax[5, 1].transAxes)
        ax[5, 5].text(x=-x_off, y=-2 * y_off, s="Ground-Truth Frames", transform=ax[5, 5].transAxes)

        plt.subplots_adjust(**SUBSIZE[dataset_name])
        plt.savefig(os.path.join(imgs_path, f"combined_img_{i+1}.png"))
        fig.clear()

    return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_dir, checkpoint, _ = get_fig_generator_arguments()
    logger = Logger(exp_path=exp_dir)
    logger.log_info("Assembling nice figures and GIFs", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Figure Generation Procedure...")
    main(exp_path=exp_dir, checkpoint=checkpoint)
