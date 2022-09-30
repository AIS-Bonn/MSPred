"""
Training and Validation of a model
"""

import os
import copy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.logger import Logger, print_, log_info
from lib.losses import KLLoss, PixelLoss, HeatmapLoss
from lib.schedulers import BetaScheduler
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
from lib.visualizations import visualize_preds, visualize_hierarch_preds
import data
from CONFIG import LOSSES


class Trainer:
    """
    Class for training and validating a model
    """

    def __init__(self, exp_path, checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        utils.set_random_seed()
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.model_name = self.exp_params["model"]["model_type"]
        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        utils.create_directory(self.plots_path, dir_name="gifs")
        utils.create_directory(self.plots_path, dir_name="images")
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"].get("shuffle_train", True)
        shuffle_eval = self.exp_params["dataset"].get("shuffle_eval", False)

        # some online updates for hierarchical models
        exp_params = copy.deepcopy(self.exp_params)
        if ("Hierarch" in self.model_name):
            # repeating the num_preds, in case only an integer was given
            periods = exp_params["model"]["HierarchLSTM"]["periods"]
            if not isinstance(self.exp_params["training"]["num_preds"], list):
                num_preds = self.exp_params["training"]["num_preds"]
                self.exp_params["training"]["num_preds"] = [num_preds] * len(periods)
            exp_params["training"]["num_preds"] = max(np.array(num_preds) * np.array(periods))

        train_set = data.load_data(exp_params=exp_params, split="train")
        valid_set = data.load_data(exp_params=exp_params, split="val")
        print_(f"  --> Number of training sequences: {len(train_set)}")
        print_(f"  --> Number of validation sequences: {len(valid_set)}")

        # setting up the extra outputs
        aux_outputs = self.exp_params["model"]["HierarchLSTM"]["aux_outputs"]
        self.aux_outputs = ("Hierarch" in self.model_name) and aux_outputs
        if aux_outputs:
            n_hmap_channels = train_set.get_num_hmap_channels()
            self.exp_params["model"]["HierarchLSTM"]["n_hmap_channels"] = n_hmap_channels
        self.dset_struct_type = train_set.get_struct_type()

        self.train_loader = data.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.valid_loader = data.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        epoch = 0
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        model = setup_model.setup_model(exp_params=self.exp_params)
        utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)

        # loading optimizer, scheduler and loss
        optimizer, scheduler = setup_model.setup_optimization(exp_params=self.exp_params, model=model)

        # loading pretrained model and other necessary objects for resuming training or fine-tuning
        if self.checkpoint is not None:
            print_(f"  --> Loading pretrained parameters from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
            if self.resume_training:
                model, optimizer, scheduler, epoch = loaded_objects
                print_(f"  --> Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects
        self.model, self.optimizer, self.scheduler, self.epoch = model, optimizer, scheduler, epoch

        # beta-scheduling
        self.beta_scheduler = BetaScheduler(
                beta_init=self.exp_params["loss"]["beta0"],
                beta_final=self.exp_params["loss"]["beta"],
                warmup_steps=self.exp_params["loss"]["beta_warmup_steps"]
            )
        self.beta = self.exp_params["loss"]["beta0"]

        # loss functions
        self.alphas = self.exp_params["loss"]["alphas"]
        self.hier_losses = []
        loss_types = self.exp_params["loss"]["reconst_losses"]
        for loss_type in loss_types:
            if loss_type not in LOSSES:
                raise ValueError(f"Unknown loss {loss_type}. Use one of {LOSSES}")
            if loss_type != "weighted_kpoint_loss":
                class_weights = None
                if self.exp_params["dataset"]["dataset_name"] == "Synpick":
                    class_weights = torch.ones((23)).float().to(self.device)
                    class_weights[-1] = 2.  # gripper
                    class_weights[0] = 0.5  # backgnd
                self.hier_losses.append(PixelLoss(loss_type, class_weights))
            else:
                w_easy_kpts = self.exp_params["loss"]["w_easy_kpts"]
                hmap_weights = self.train_loader.dataset.get_heatmap_weights(w_easy_kpts=w_easy_kpts)
                self.hier_losses.append(HeatmapLoss(hmap_weights))
        self.kl_loss = KLLoss()
        return

    def get_total_loss(self, mse_loss, kl_loss):
        """ Weighting and computing the final loss value"""
        if not isinstance(mse_loss, dict):
            loss = mse_loss
        else:
            loss = 0.0
            for h, mse in mse_loss.items():
                loss += (self.alphas[h] * mse)
        loss += self.beta * kl_loss
        return loss

    @emergency_save
    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss/Comb_loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )
            self.writer.add_scalar('Learning/LR', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Learning/KL-Loss Beta', self.beta, epoch)

            # updating beta and learning rate schedulers
            self.beta = self.beta_scheduler.step(iter=epoch)
            setup_model.update_scheduler(
                    scheduler=self.scheduler,
                    exp_params=self.exp_params,
                    control_metric=self.validation_losses[-1],
                    iter=epoch,
                    end_epoch=True
                )

            # saving backup model checkpoint and (if reached saving frequency) epoch checkpoint
            setup_model.save_checkpoint(  # Gets overriden every epoch: checkpoint_last_saved.pth
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    exp_path=self.exp_path,
                    savedir="models",
                    savename="checkpoint_last_saved.pth"
                )
            if(epoch % save_frequency == 0 and epoch != 0):  # checkpoint_epoch_XX.pth
                print_(f"Saving model checkpoint: {epoch}")
                setup_model.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        exp_path=self.exp_path,
                        savedir="models"
                    )

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        setup_model.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir="models",
                finished=True
            )
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        # training parameters
        tf_epochs = self.exp_params["training"]["tf_epochs"]
        context = self.exp_params["training"]["context"]
        num_preds = self.exp_params["training"]["num_preds"]
        num_iters = min(self.exp_params["training"]["num_iters"], len(self.train_loader))
        teacher_force = (epoch <= tf_epochs)

        # initializinng losses
        epoch_losses, kl_losses = [], []
        mse_losses = defaultdict(list) if self.aux_outputs else []

        # training epoch
        progress_bar = tqdm(enumerate(self.train_loader), total=num_iters)
        for i, inputs_ in progress_bar:
            iter_ = num_iters * epoch + i

            # forward pass
            frames = inputs_["frames"]
            heatmaps = inputs_["heatmaps"] if "heatmaps" in inputs_ else []
            batch_classes = inputs_["classes"] if "classes" in inputs_ else []
            frames = frames.to(self.device)
            out_dict = self.model(
                    x=frames,
                    context=context,
                    num_preds=num_preds,
                    teacher_force=teacher_force
                )

            # assembling targets
            targets = [frames.float()]
            if self.aux_outputs:
                for hmap in heatmaps:
                    targets.append(hmap.to(self.device))

            # computing loss values
            # kl-loss
            kl_loss = self.kl_loss(
                    mu1=out_dict["mu_post"], logvar1=out_dict["logvar_post"],
                    mu2=out_dict["mu_prior"], logvar2=out_dict["logvar_prior"]
                )
            kl_losses.append(kl_loss.item())
            # losses for each decoder-head
            if self.aux_outputs:
                mse_loss_dict = {}
                for h, preds in out_dict["preds"].items():
                    mse_loss = self.hier_losses[h](
                            preds,
                            targets[h][:, out_dict["target_masks"][h]],
                            batch_classes
                        )
                    mse_loss_dict[h] = mse_loss
                    mse_losses[h].append(mse_loss.item())
                loss = self.get_total_loss(mse_loss_dict, kl_loss)
            # loss only at frame level
            else:
                mse_loss = self.hier_losses[0](
                        preds=out_dict["preds"],
                        targets=targets[0][:, out_dict["target_masks"]]
                    )
                mse_losses.append(mse_loss.item())
                loss = self.get_total_loss(mse_loss, kl_loss)
            epoch_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logging to tensorboard every once in a while
            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                if self.aux_outputs:
                    for h, losses in mse_losses.items():
                        self.writer.add_scalar(
                                name=f"Loss/Train_Loss_Level_{h}",
                                val=np.mean(mse_losses[h]),
                                step=iter_
                            )
                else:
                    self.writer.add_scalar(
                            name="Loss/MSE_Train_Loss",
                            val=np.mean(mse_losses),
                            step=iter_
                        )
                self.writer.add_scalar("Loss/KL_Train_Loss", np.mean(kl_losses), iter_)
                self.writer.add_scalar("Loss/Train_Loss", np.mean(epoch_losses), iter_)
                log_data = f"""Log data train iteration {iter_}:  loss={round(np.mean(epoch_losses), 5)};"""
                log_info(message=log_data)

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        self.training_losses.append(np.mean(epoch_losses))
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        # validation parameters
        context = self.exp_params["training"]["context"]
        num_preds = self.exp_params["training"]["num_preds"]
        num_iters = len(self.valid_loader)

        # initializinng losses
        epoch_losses, kl_losses = [], []
        mse_losses = defaultdict(list) if self.aux_outputs else []

        # training epoch
        progress_bar = tqdm(enumerate(self.valid_loader), total=num_iters)
        for i, inputs_ in progress_bar:
            # forward pass
            frames = inputs_["frames"]
            heatmaps = inputs_["heatmaps"] if "heatmaps" in inputs_ else []
            batch_classes = inputs_["classes"] if "classes" in inputs_ else []
            frames = frames.to(self.device)
            out_dict = self.model(
                    x=frames,
                    context=context,
                    num_preds=num_preds,
                    teacher_force=False
                )

            # assembling targets
            targets = [frames.float()]
            if self.aux_outputs:
                for hmap in heatmaps:
                    targets.append(hmap.to(self.device))

            # computing loss values
            # kl-loss
            kl_loss = self.kl_loss(
                    mu1=out_dict["mu_post"], logvar1=out_dict["logvar_post"],
                    mu2=out_dict["mu_prior"], logvar2=out_dict["logvar_prior"]
                )
            kl_losses.append(kl_loss.item())
            # losses for each decoder-head
            if self.aux_outputs:
                mse_loss_dict = {}
                for h, preds in out_dict["preds"].items():
                    mse_loss = self.hier_losses[h](
                            preds,
                            targets[h][:, out_dict["target_masks"][h]],
                            batch_classes
                        )
                    mse_loss_dict[h] = mse_loss
                    mse_losses[h].append(mse_loss.item())
                loss = self.get_total_loss(mse_loss_dict, kl_loss)
            # loss only at frame level
            else:
                mse_loss = self.hier_losses[0](
                        preds=out_dict["preds"],
                        targets=targets[0][:, out_dict["target_masks"]]
                    )
                mse_losses.append(mse_loss.item())
                loss = self.get_total_loss(mse_loss, kl_loss)
            epoch_losses.append(loss.item())

            # saving some visualizations
            if (i == 0):
                # visualizing frame predictions only
                if not self.aux_outputs:
                    preds = out_dict["preds"]
                    seq = torch.cat([frames[0][:context], preds[0]], dim=0)
                    savepath = os.path.join(self.plots_path, "images", f"img_epoch_{epoch}.png")
                    visualize_preds(sequence=seq, savepath=savepath, n_cols=8, n_seed=context)
                # visualizing frames and higher-level predictions
                else:
                    # sampling outs to show: (1, 2, ...), (t1, 2*t1, ...), (t2, 2*t2, ...)
                    frame_nums = {}
                    for h in range(len(out_dict["target_masks"])):
                        frame_nums[h] = utils.mask_to_fnums(
                                mask=out_dict["target_masks"][h][context:],
                                n_preds=num_preds[h],
                                n_seed=context
                            )
                    preds, gt_preds = {}, {}
                    for h in range(len(out_dict["preds"])):
                        preds[h] = out_dict["preds"][h][0][:num_preds[h]]
                        gt_preds[h] = targets[h][0, out_dict["target_masks"][h]][:num_preds[h]]

                    # visualizing predictions at each level
                    savepath = os.path.join(self.plots_path, "images", f"img_epoch_{epoch}.png")
                    visualize_hierarch_preds(
                            gt_seq=frames[0],
                            preds=preds,
                            fnums=frame_nums,
                            n_seed=context,
                            savepath=savepath,
                            struct_types=self.dset_struct_type
                        )
                    # visualizing grount-truth targets at each level
                    savepath = os.path.join(self.plots_path, "images", f"img_epoch_{epoch}_gt.png")
                    visualize_hierarch_preds(
                            gt_seq=frames[0],
                            preds=gt_preds,
                            fnums=frame_nums,
                            n_seed=context,
                            savepath=savepath,
                            struct_types=self.dset_struct_type
                        )
            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

        # logging average validation losses
        if self.aux_outputs:
            for h, losses in mse_losses.items():
                self.writer.add_scalar(
                        name=f"Loss/Valid_Loss_Level_{h}",
                        val=np.mean(mse_losses[h]),
                        step=epoch
                    )
        else:
            self.writer.add_scalar(
                    name="Loss/Valid_Loss_MSE",
                    val=np.mean(mse_losses),
                    step=epoch
                )
        self.writer.add_scalar("Loss/KL_Valid_Loss", np.mean(kl_losses), epoch)
        self.writer.add_scalar("Loss/Valid_Loss", np.mean(epoch_losses), epoch)
        log_data = f"""Log data valid epoch {epoch}: loss={round(np.mean(epoch_losses), 5)};"""
        log_info(message=log_data)
        self.validation_losses.append(np.mean(epoch_losses))
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting training procedure", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Trainer...")
    trainer = Trainer(exp_path=exp_path, checkpoint=args.checkpoint, resume_training=args.resume_training)
    print_("Loading dataset...")
    trainer.load_data()
    print_("Setting up model and optimizer")
    trainer.setup_model()
    print_("Starting to train")
    trainer.training_loop()


#
