"""
Evaluating a model checkpoint
"""

import os
import copy
from tqdm import tqdm
import numpy as np
import torch

from lib.arguments import get_eval_arguments
from lib.config import Config
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class Evaluator:
    """ Class for evaluating a model """

    def __init__(self, exp_path, checkpoint, n_samples_per_seq=1):
        """ Initializing the evaluator object """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint
        self.checkpoint_name = os.path.basename(checkpoint)
        self.n_samples_per_seq = n_samples_per_seq

        # creating results directories
        model_type = self.exp_params["model"]["model_type"]
        self.model_type = model_type
        self.models_path = os.path.join(self.exp_path, "models")
        self.results_dir = os.path.join(exp_path, "results")
        utils.create_directory(self.results_dir)
        utils.create_directory(self.results_dir, dir_name=f"results_{self.checkpoint_name}")
        self.plots_path = os.path.join(self.results_dir, f"plots_{checkpoint}")
        utils.create_directory(self.plots_path)
        utils.create_directory(self.plots_path, dir_name="gifs")
        utils.create_directory(self.plots_path, dir_name="images")
        return

    def load_data(self):
        """ Loading dataset and fitting data-loader for iterating in a batch-like fashion """
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

        # data loader
        self.test_loader = data.build_data_loader(
                dataset=self.test_set,
                batch_size=self.batch_size,
                shuffle=True
            )
        return

    def setup_model(self):
        """ Initializing model and loading pretrained paramters """
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

        # initializing metrcic trackers
        num_hierarch = self.exp_params["model"]["HierarchLSTM"]["num_hierarch"]
        self.metricTrackers = {}
        num_hier = 1 if not self.aux_outputs else num_hierarch
        for h in range(num_hier):
            if h == 0:
                metrics = self.test_set.METRICS_LEVEL_0
            elif h == 1:
                metrics = self.test_set.METRICS_LEVEL_1
            elif h == 2:
                metrics = self.test_set.METRICS_LEVEL_2
            else:
                raise ValueError(f"Wrong hierarch level {h}. Must be one of (0, 1, 2)...")
            self.metricTrackers[h] = MetricTracker(metrics=metrics)
        return

    @torch.no_grad()
    def evaluate(self):
        """ Evaluating model epoch loop """
        # evaluation parameters and helpers
        context = self.exp_params["eval"]["context"]
        num_preds = self.exp_params["eval"]["num_preds"]
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))

        # evaluation
        for i, inputs_ in progress_bar:
            # preparing inputs and targets
            frames = inputs_["frames"]
            heatmaps = inputs_["heatmaps"] if "heatmaps" in inputs_ else []
            frames = frames.to(self.device)
            targets = [frames.float()]
            if self.aux_outputs:
                for hmap in heatmaps:
                    targets.append(hmap.to(self.device).float())

            # repeating N times, though sampling different latents each time
            for s in range(self.n_samples_per_seq):
                # forward pass
                out_dict = self.model(
                        x=frames,
                        context=context,
                        num_preds=num_preds,
                        teacher_force=False,
                    )

                # computing metrics
                frame_nums = {}
                for h, preds in out_dict["preds"].items():
                    if not self.aux_outputs and h > 0:
                        break
                    self.metricTrackers[h].accumulate(
                            preds=preds,
                            targets=targets[h][:, out_dict["target_masks"][h]]
                        )
                    frame_nums[h] = utils.mask_to_fnums(
                            mask=out_dict["target_masks"][h][context:],
                            n_preds=num_preds[h],
                            n_seed=context
                        )

            # using best result among all sampled vectors
            for metricTracker in self.metricTrackers.values():
                metricTracker.find_best_sample_results(num_last_results=self.n_samples_per_seq)
            progress_bar.set_description(f"Evaluation iter {i}")

        # saving results
        for h, metricTracker in self.metricTrackers.items():
            file = os.path.join(self.results_dir, f"results_{self.checkpoint_name}", f"metrics_{h}.json")
            metricTracker.aggregate()
            _ = metricTracker.summary()
            metricTracker.save_results(results_file=file, summary=False)
            metricTracker.save_plots(
                    savepath=self.plots_path,
                    suffix=f"_level_{h}",
                    start_frame=context,
                    frame_nums=frame_nums[h]
                )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_eval_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting evaluation procedure", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Evaluator...")
    evaluator = Evaluator(
            exp_path=exp_path,
            checkpoint=args.checkpoint,
            n_samples_per_seq=args.n_samples_per_seq
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and loading pretrained parameters")
    evaluator.setup_model()
    print_("Starting evaluation")
    evaluator.evaluate()


#
