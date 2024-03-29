import torch
from abc import abstractmethod
from numpy import inf
from utils.log import WandB

"""
- Training process logging
- Checkpoint saving
- Checkpoint resuming
- Reconfigurable performance monitoring for saving current best model, and early stop training.
    - If config monitor is set to max val_accuracy, which means then the trainer will save a checkpoint model_best.pth 
      when validation accuracy of epoch replaces current maximum.
    - If config early_stop is set, training will be automatically terminated when model performance does not improve for given number of 
    epochs. This feature can be turned off by passing 0 to the early_stop option, or just deleting the line of config.
"""


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model,
        config,
        data_loader,
        len_epoch=None,
    ):
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 0
        self.iters = 0
        self.checkpoint_dir = config.save_dir

        # Setup visualization writer instance

        if cfg_trainer["visual_tool"] == "wandb":
            visual_config = {
                "Architecture": config["arch"]["type"],
                "trainer": cfg_trainer["type"],
            }
            self.track = WandB(
                config["name"],
                cfg_trainer,
                self.logger,
                cfg_trainer["visual_tool"],
                visualize_config=visual_config,
            )

        elif cfg_trainer["visual_tool"] == "None":
            self.track = None

        else:
            raise ImportError(
                "Visualization tool isn't exists, please refer to comment 1.* "
                "to choose appropriate module"
            )

        self.meta_vocab_size = config["arch"]["args"]["vocab_size"]

        """
        if config.init_from == "resume":
            print("Resuming training from output")
            self._resume_checkpoint(config.init_from)

        
        elif config.init_from == "scratch":
            print("Initializing a new model from scratch")

            # determine the vocab size we'll use for from-scratch training
            if self.meta_vocab_size is None:
                print(
                    "defaulting to vocab_size of VinAI-BART-Pho to 40031 (40031 rounded up for efficiency)"
                )

            model_args["vocab_size"] = (
                self.meta_vocab_size if self.meta_vocab_size is not None else 40031
            )

            model = GPT2(GPTConfig(**model_args))
        """

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        running_mfu = -1.0
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # Logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # Logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # Evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        # self.track: WandB Class  -> self.track.write: WandB Library
        # Launch multiple runs from one script?​
        # run.finish(): Use this at the end of your run to finish logging for that run
        if self.track is not None and self.track.name == "wandb":
            self.track.writer.finish()

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "iter": self.iters,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # crop down the model block size if desired, using model surgery
        if block_size < model.config.block_size:
            self.model.crop_block_size(block_size)
            model_args[
                "block_size"
            ] = block_size  # so that the checkpoint will have the right value

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
