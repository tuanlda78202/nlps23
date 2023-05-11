# ENV
import os, logging

os.environ["WANDB_SILENT"] = "True"
os.environ["WANDB_MODE"] = "offline"
os.makedirs("out", exist_ok=True)
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)

import numpy as np
import torch
from contextlib import nullcontext
from scripts import BaseTrainer
from utils import 
from tqdm import tqdm
import wandb

""" 
{'name': 'VietnamesePoem-GPT2-124M', 'data': {'dataset': 'vietnamese-poem', 'gradient_accumulation_steps': '5 * 1', 
'batch_size': 12, 'block_size': 1024}, 'arch': {'type': 'GPT2', 'args': {'n_layer': 12, 'n_head': 12, 'n_emb': 768, 
'dropout': 0.0, 'bias': False}}, 'optimizer': {'type': 'LRDecay', 'args': {'decay_lr': True, 'warmup_iters': 2000, 
'lr_decay_iters': 600000, 'min_lr': '6e-5'}}, 'eval': {'eval_interval': 2000, 'eval_iters': 200, 'log_interval': 10, 
'eval_only': False, 'always_save_checkpoint': True, 'init_from': 'scratch'}, 'system': {'device': 'cuda', 'dtype': 'bfloat16', 
'compile': True}, 'wandb': {'wandb_log': False, 'wandb_project': 'nlps23', 'wandb_run_name': 'vietnamese-poem_gpt2-124M'}}
"""


class GPT2Trainer(BaseTrainer):
    """
    GPT2 Trainer class
    """

    def __init__(
        self,
        model,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        len_epoch=None,
    ):
        super().__init__(model, config, data_loader)

        self.config = config
        self.data_config = self.config["data"]

        self.device = device
        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16)
        )
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)

        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # DataFrame metrics
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], track=self.track
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], track=self.track
        )

        # GPT2
        tokens_per_iter = (
            self.data_config["gradient_accumulation_steps"]
            * self.data_config["batch_size"]
            * self.data_config["block_size"]
        )
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        tqdm_batch = tqdm(
            iterable=self.data_loader,
            desc="Epoch {}".format(epoch),
            total=len(self.data_loader),
            unit="it",
        )

        self.model.train()
        self.train_metrics.reset()

        # FWD&BWD update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16

        for micro_step in range(self.data_config["gradient_accumulation_steps"]):
            with self.ctx:
                logits, loss = self.model(X, Y)
                loss = (
                    loss / self.data_config["gradient_accumulation_steps"]
                )  # scale the loss to account for gradient accumulation

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # Clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        """
        for batch_idx, loader in enumerate(tqdm_batch):
            # Load to Device
            if self.device == "cuda":
                data = loader["img"].to(
                    device=self.device, dtype=torch.cuda.FloatTensor
                )
                mask = loader["mask"].to(
                    device=self.device, dtype=torch.cuda.FloatTensor
                )

            else:
                data = loader["img"].to(device=self.device)
                data = data.type(torch.FloatTensor)
                mask = loader["mask"].to(device=self.device)
                mask = mask.type(torch.FloatTensor)

            self.optimizer.zero_grad()

            # x_map for metrics, list_maps for loss
            x_map, list_maps = self.model(data)

            loss = self.criterion(list_maps, mask)
            loss.backward()
            self.optimizer.step()

            # Variable for logging
            log_loss = loss.item()

            # Metrics, detach tensor auto-grad to numpy
            if self.device == "cuda":
                map_np, mask_np = (
                    x_map.cpu().detach().numpy(),
                    mask.cpu().detach().numpy(),
                )
            else:
                map_np, mask_np = x_map.detach().numpy(), mask.detach().numpy()

            # Metrics
            log_mae = mae(map_np, mask_np)
            log_sm = sm(map_np, mask_np)

            # Progress bar
            tqdm_batch.set_postfix(loss=log_loss, mae=log_mae, sm=log_sm)

            # WandB
            wandb.log({"loss": log_loss, "mae": log_mae, "sm": log_sm})

            # Logging
            self.track.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", log_loss)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(map_np, mask_np))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

            if batch_idx == self.len_epoch:
                break
        """
        tqdm_batch.close()

        log = self.train_metrics.result()

        if iter_num % eval_interval == 0 and self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with torch.no_grad():
            losses = {}

            for split in ["train", "val"]:
                losses = torch.zeros(eval_iters)

                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    with ctx:
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()

                losses[split] = losses.mean()

            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )

        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

            for batch_idx, loader in enumerate(self.valid_data_loader):
                # Load to Device
                if self.device == "cuda":
                    data = loader["img"].to(
                        device=self.device, dtype=torch.cuda.FloatTensor
                    )
                    mask = loader["mask"].to(
                        device=self.device, dtype=torch.cuda.FloatTensor
                    )

                elif self.device == "cpu":
                    data = loader["img"].to(device=self.device, dtype=torch.FloatTensor)
                    mask = loader["mask"].to(
                        device=self.device, dtype=torch.FloatTensor
                    )

                else:  # MPS
                    data = loader["img"].to(device=self.device, dtype=torch.float32)
                    mask = loader["mask"].to(device=self.device, dtype=torch.float32)

                # Forward
                x_map, list_maps = self.model(data)
                loss = self.criterion(list_maps, mask)

                # Metrics, detach tensor auto-grad to numpy
                if self.device == "cuda":
                    map_np, mask_np = (
                        x_map.cpu().detach().numpy(),
                        mask.cpu().detach().numpy(),
                    )
                else:
                    map_np, mask_np = x_map.detach().numpy(), mask.detach().numpy()

                # Logging
                self.track.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(map_np, mask_np))

                # Log WandB, Predicted will show first
                images = wandb.Image(make_grid(x_map[:8], nrow=4))
                self.track.log({"Predicted": images}, step=None)

                # Delete garbage
                del images
                gc.collect()

        # WandB Log Original + GT
        loader = next(iter(self.data_loader))

        self.track.set_step(epoch, "valid")

        # Grid 2 x 4
        original = wandb.Image(make_grid(loader["img"][:8], nrow=4))
        gt = wandb.Image(make_grid(loader["mask"][:8], nrow=4))

        self.track.log({"Original": original}, step=None)
        self.track.log({"Ground Truth": gt}, step=None)

        # Delete garbage
        del original, gt
        gc.collect()

        # Add histogram of model parameters to the WandB
        for name, p in self.model.named_parameters():
            self.track.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
