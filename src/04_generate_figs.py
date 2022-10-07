"""
Generating figures and gifs for a particular video prediction dataset
using a pretrained model checkpoint
"""

import os
from tqdm import tqdm
import copy
import numpy as np
import torch

from lib.arguments import get_fig_generator_arguments
from lib.config import Config
from lib.logger import Logger, print_
import lib.setup_model as setup_model
import lib.utils as utils
import lib.visualizations as visualizations
import data


class FigGenerator:
    """
    Main module for figure generation:
        1: Loading pretrained model
        2: Loading test dataset
        3: Inference of some images
        4: Saving predictions and ground truth

    Args:
    -----
    exp_path: string
        path to the experiment directory
    checkpoint: string
        Name of the model checkpoint to use to generate figures
    num_imgs: integer
        Number of images and gifs to generate
    """

    def __init__(self, exp_path, checkpoint, num_imgs=5):
        """ Module initializer """
        # configs
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()

        # relevan variables
        self.checkpoint = checkpoint
        self.checkpoint_name = os.path.basename(checkpoint)
        self.num_imgs = num_imgs
        model_type = self.exp_params["model"]["model_type"]
        self.model_type = model_type

        # paths
        self.models_path = os.path.join(self.exp_path, "models")
        self.results_dir = os.path.join(exp_path, "results")
        self.plots_path = os.path.join(self.results_dir, f"plots_{checkpoint}")
        self.imgs_path = os.path.join(self.plots_path, "images")
        self.gif_resources_path = os.path.join(self.plots_path, "gif_resources")
        self.gifs_path = os.path.join(self.plots_path, "gifs")
        utils.create_directory(self.imgs_path)
        utils.create_directory(self.gif_resources_path)
        utils.create_directory(self.gifs_path)
        return

    def load_data(self):
        """ Loading dataset """
        self.batch_size = self.exp_params["eval"]["batch_size"]
        aux_outputs = self.exp_params["model"]["HierarchLSTM"]["aux_outputs"]
        self.aux_outputs = ("Hierarch" in self.model_type) and aux_outputs

        # loading test set
        exp_params = copy.deepcopy(self.exp_params)
        if ("Hierarch" in self.model_type):
            # repeating the num_preds, in case only an integer was given
            periods = exp_params["model"]["HierarchLSTM"]["periods"]
            if not isinstance(self.exp_params["eval"]["num_preds"], list):
                num_preds = self.exp_params["eval"]["num_preds"]
                self.exp_params["eval"]["num_preds"] = [num_preds] * len(periods)
            exp_params["eval"]["num_preds"] = max(np.array(num_preds) * np.array(periods))
        self.test_set = data.load_data(exp_params=exp_params, split="test")
        print_(f"  --> Number of test sequences: {len(self.test_set)}")

        # setting up data for higher-levels
        n_hmap_channels = self.test_set.get_num_hmap_channels()
        if aux_outputs:
            self.exp_params["model"]["HierarchLSTM"]["n_hmap_channels"] = n_hmap_channels
        self.dset_struct_type = self.test_set.get_struct_type()

        self.test_loader = data.build_data_loader(
                dataset=self.test_set,
                batch_size=self.batch_size,
                shuffle=True
            )
        return

    def load_model(self):
        """ Loading pretrained model """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model and pretrained parameters
        model = setup_model.setup_model(exp_params=self.exp_params)
        checkpoint_path = os.path.join(self.models_path, self.checkpoint)
        model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                only_model=True
            )
        self.model = model.eval().to(self.device)
        self.model = self.model.train()
        return

    @torch.no_grad()
    def generate_figs(self):
        """ Obtaining predictions and saving figures """
        context = self.exp_params["eval"]["context"]
        num_preds = self.exp_params["eval"]["num_preds"]

        iterator = iter(self.test_loader)
        for i in tqdm(range(self.num_imgs)):
            inputs_ = next(iterator)
            frames = inputs_["frames"]
            heatmaps = inputs_["heatmaps"] if "heatmaps" in inputs_ else []
            frames = frames.to(self.device)
            targets = [frames.float()]
            if self.aux_outputs:
                for hmap in heatmaps:
                    targets.append(hmap.to(self.device).float())

            # forward pass
            out_dict = self.model(
                    x=frames,
                    context=context,
                    num_preds=num_preds,
                    teacher_force=False
                )

            # VISUALIZATIONS
            # basic video-pred
            if not self.aux_outputs:
                frame_nums_vis = {0: utils.mask_to_fnums(
                            mask=out_dict["target_masks"][0][context:],
                            n_preds=num_preds[0],
                            n_seed=context
                        )
                    }
                preds = out_dict["preds"]
                seq = torch.cat([frames[0][:context], preds[0]], dim=0)
                savepath = os.path.join(self.imgs_path, f"img_{i+1}.png")
                visualizations.visualize_preds(
                        sequence=seq,
                        savepath=savepath,
                        n_cols=8,
                        n_seed=context
                    )
            # hiearchical visualizations + basic video-pred
            else:
                # Plot predictions
                frame_nums_vis = {h: utils.mask_to_fnums(
                            mask=out_dict["target_masks"][h][context:],
                            n_preds=num_preds[h],
                            n_seed=context
                        ) for h in range(len(out_dict["target_masks"]))
                    }
                preds = {h: out_dict["preds"][h][0] for h in range(len(out_dict["preds"]))}
                savepath = os.path.join(self.imgs_path, f"img_{i+1}.png")
                seq_dir = os.path.join(self.gif_resources_path, f"seq_{i+1}")
                visualizations.visualize_hierarch_preds(
                        gt_seq=frames[0],
                        preds=preds,
                        fnums=frame_nums_vis,
                        n_seed=context,
                        seq_dir=seq_dir,
                        save_frames=True,
                        savepath=savepath,
                        struct_types=self.dset_struct_type
                    )
                # Plot Ground-truth
                gt_preds = {
                    h: targets[h][0, out_dict["target_masks"][h]] for h in range(len(out_dict["preds"]))
                }
                savepath = os.path.join(self.imgs_path, f"img_{i+1}_gt.png")
                seq_dir = os.path.join(self.gif_resources_path, f"seq_{i+1}_gt")
                visualizations.visualize_hierarch_preds(
                        gt_seq=frames[0],
                        preds=gt_preds,
                        fnums=frame_nums_vis,
                        n_seed=context,
                        seq_dir=seq_dir,
                        save_frames=True,
                        savepath=savepath,
                        struct_types=self.dset_struct_type
                    )

        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_dir, checkpoint, num_imgs = get_fig_generator_arguments()
    logger = Logger(exp_path=exp_dir)
    logger.log_info("Starting figure generation procedure", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Figure Generation Procedure...")
    figGenerator = FigGenerator(
            exp_path=exp_dir,
            checkpoint=checkpoint,
            num_imgs=num_imgs
        )
    print_("Loading data...")
    figGenerator.load_data()
    print_("Loading model...")
    figGenerator.load_model()
    print_("Generating figures...")
    figGenerator.generate_figs()
    print_("Figures and GIFs have been successfully generated...")

#
