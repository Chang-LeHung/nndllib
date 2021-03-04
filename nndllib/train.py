#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Mr. HuChang

import sys as _sys
import seaborn as _sns
import torch as _torch
import torch.nn as _nn
from tqdm.auto import tqdm as _tqdm
import prettytable as _pt


class TorchTrainer(object):

    def __init__(self):
        self.trained_epochs = 0
        self.trained_steps = 0
        self.device = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
        self._lr_scheduler_enabled = False
        self._temp_pbar = None

    def test_step(self, batch, batch_idx, optimizer_idx=None):
        pass

    def test_epoch_start(self):
        pass

    def test_epoch_end(self, outputs):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def training_epoch_start(self):
        pass

    def training_epoch_end(self, epoch_outputs):
        pass

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        pass

    def validation_epoch_start(self):
        pass

    def validation_epoch_end(self, epoch_outputs):
        pass

    def configure_optimizer(self):
        pass

    def optimizer_step(self, loss, optimizer, optimizer_idx=None):

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def on_epoch_end(self):
        pass

    def on_epoch_start(self):
        pass

    def fit(self, epochs, prog_bar_refresh_rate=1, val_every_n_epoch=1):

        self._data_init(prog_bar_refresh_rate, val_every_n_epoch)
        length = len(self._trainset)
        with _tqdm(dynamic_ncols=True, total=len(self._trainset)) as pbar:
            for i in range(epochs):
                self.training_epoch_start()
                self.on_epoch_start()
                epoch_outputs = []
                pbar.set_description_str(f"Epoch={1 + self.trained_epochs} step[{i + 1}/{epochs}]")
                for idx, batch in enumerate(self._trainset):
                    batch = self._type_transfer(batch)
                    for optimizer_idx in range(len(self._optimizers)):
                        step_out = self.training_step(batch, idx, optimizer_idx)
                        if step_out is None:
                            break
                        epoch_outputs.append(step_out)
                        loss = None
                        if isinstance(step_out, dict):
                            loss = step_out['loss']
                        elif isinstance(step_out, _torch.Tensor):
                            loss = step_out
                        else:
                            raise TypeError("what training_step return should be Tensor or dict with key=loss with and its value type is Tensor")
                        self.optimizer_step(loss, self._optimizers[optimizer_idx], optimizer_idx=optimizer_idx)
                    self.trained_steps += 1

                    self._bar_show(pbar)
                    if (idx + 1) % self.prog_bar_refresh_rate == 0 or idx + 1 == length:
                        if idx - 1 == length:
                            pbar.update(idx % self.prog_bar_refresh_rate + 1)
                        else:
                            pbar.update(self.prog_bar_refresh_rate)
                            
                for optimizer_idx in range(len(self._optimizers)):
                    if self._lr_scheduler_enabled and len(self._lr_schedulers) >= optimizer_idx:
                        self._lr_schedulers[optimizer_idx].step()
                        
                if i < epochs - 1:
                    pbar.update(-length)
                self.training_epoch_end(epoch_outputs)


                if (i + 1) % self.val_every_n_epoch == 0 and self._validation_enbale:
                    epoch_outputs = []
                    self._before_validation_epoch_start()
                    self.validation_epoch_start()
                    with _torch.no_grad():
                        with _tqdm(dynamic_ncols=True, total=len(self._valset)) as bar:
                            bar.set_description(f"Epoch={self.trained_epochs} Validating")
                            for idx, batch in enumerate(self._valset):
                                batch = self._type_transfer(batch)
                                for optimizer_idx in range(len(self._optimizers)):
                                    step_out = self.validation_step(batch, idx, optimizer_idx)
                                    epoch_outputs.append(step_out)
                                    loss = None
                                    if isinstance(step_out, dict):
                                        loss = step_out['loss']
                                    elif isinstance(step_out, _torch.Tensor):
                                        loss = step_out
                                    else:
                                        raise TypeError("what test_step return should be Tensor or dict with key=loss with and its value type is Tensor")
                                bar.set_postfix_str("loss={:.3}".format(loss.item()))
                                bar.update(1)
                            bar.set_description(f"Epoch={self.trained_epochs + 1} Validated")
                    self.validation_epoch_end(epoch_outputs)
                    self._after_validation_epoch_end()

                self.on_epoch_end()
                self.trained_epochs += 1

    def test(self, test_loader=None):

        if test_loader is not None:
            self._testset = test_loader

        with _torch.no_grad():
            self._before_test_epoch_start()
            self.test_epoch_start()
            epoch_outputs = []
            with _tqdm(dynamic_ncols=True, total=len(self._testset)) as bar:
                bar.set_description(f"Epoch={self.trained_epochs} Testing")
                for idx, batch in enumerate(self._testset):
                    batch = self._type_transfer(batch)
                    for optimizer_idx in range(len(self._optimizers)):
                        step_out = self.test_step(batch, idx, optimizer_idx)
                        epoch_outputs.append(step_out)
                        loss = None
                        if isinstance(step_out, dict):
                            loss = step_out['loss']
                        elif isinstance(step_out, _torch.Tensor):
                            loss = step_out
                        else:
                            raise TypeError(
                                "what training_step return should be Tensor or dict with key=loss with and its value type is Tensor")
                    bar.set_postfix_str("loss={:.3}".format(loss.item()))
                    bar.update(1)
                bar.set_description(f"Epoch={self.trained_epochs} Tested")

            self.test_epoch_end(epoch_outputs)
            return self._after_test_epoch_end(epoch_outputs)

    def _bar_show(self, bar):
        show_str = ""
        for key, value in self._temp_pbar.items():
            show_str += "{}={:.3} ".format(key, float(value))
        bar.set_postfix_str(show_str)

    def log(self, name, value=None):
        if self._temp_pbar is None:
            self._temp_pbar = dict()
        if value is None:
            self._temp_pbar = {**self._temp_pbar, **name}
        else:
            self._temp_pbar[name] = value

    def val_dataloader(self):
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def _data_init(self, prog_bar_refresh_rate, val_every_n_epoch):

        self.prog_bar_refresh_rate = prog_bar_refresh_rate
        self.val_every_n_epoch = val_every_n_epoch
        optims = self.configure_optimizer()

        self._optimizer_lr_scheduler_check()

        tb = _pt.PrettyTable()
        tb.field_names = ["Num", "Name", "Type", "Trainable Params", "Non-Trainable Params"]
        idx = 0
        train_sum = 0
        non_sum = 0
        for name in dir(self):
            grad_true = 0
            grad_false = 0
            if not (name.startswith('__') and name.endswith("__")):
                item = getattr(self, name)
                if isinstance(item, _nn.Module):
                    setattr(self, name, item.to(self.device))
                    grad_true = sum([m.numel() for m in item.parameters() if m.requires_grad])
                    grad_false = sum([m.numel() for m in item.parameters() if not m.requires_grad])
                    train_sum += grad_true
                    non_sum += grad_false
                    tb.add_row([idx, name, item.__class__.__name__,
                                str(grad_true // 1000) + " K", str(grad_false // 1000) + " K"])
                    idx += 1

        print(tb, file=_sys.stderr)

        print(str(train_sum // 1000) + " K\t", "Trainable params", file=_sys.stderr)
        print(str(non_sum // 1000) + " K\t", "Non-Trainable params", file=_sys.stderr)
        print(str((non_sum + train_sum) // 1000) + " K\t", "Total params", file=_sys.stderr)
        self._trainset = self.train_dataloader()

        if not "TorchTrainer" in getattr(self, "validation_step").__str__():
            self._validation_enbale = True
            if "TorchTrainer" in getattr(self, "val_dataloader").__str__():
                raise RuntimeError("please override funciton val_dataloader")
            self._valset = self.val_dataloader()

        else:
            self._validation_enbale = False

        if not "TorchTrainer" in getattr(self, "test_step").__str__():
            if "TorchTrainer" in getattr(self, "test_dataloader").__str__():
                raise RuntimeError("please override funciton test_dataloader")
            self._testset = self.test_dataloader()


    def _before_test_epoch_start(self):
        self._before_validation_epoch_start()

    def _after_test_epoch_end(self, outputs):
        self._after_validation_epoch_end()
        # if return sth output GPU Memory will cost twice
        # return outputs

    def _before_validation_epoch_start(self):

        self._change_model_state(False)

    def _after_validation_epoch_end(self):
        self._change_model_state(True)

    def _change_model_state(self, train=False):
        for name in dir(self):
            if not (name.startswith("__") and name.endswith("__")):
                var = getattr(self, name)
                if isinstance(var, _nn.Module):
                    if train:
                        var.train()
                    else:
                        var.train(False)
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    def _optimizer_lr_scheduler_check(self):

        returns = self.configure_optimizer()

        if isinstance(returns, _torch.optim.Optimizer):
            self._optimizers = [returns]
            return None

        if isinstance(returns, (tuple, list)):

            if all([isinstance(item, _torch.optim.Optimizer) for item in returns]):
                self._optimizers = returns
                return None

            if all([isinstance(item, (list, tuple)) for item in returns]) and len(returns) == 2:
                self._optimizers = returns[0]
                self._lr_schedulers = returns[1]
                self._lr_scheduler_enabled = True
                return None

        raise RuntimeError("configure_optimizer returns error, which should be optimizer or\
                         optimizer1, optimizer2, ...or [optimizer1, optimizer2, ...],\
                          [lr_scheduler1, lr_scheduler2, ...]")

    def _type_transfer(self, batch):

        if isinstance(batch, _torch.Tensor):
            return batch.to(self.device)

        if isinstance(batch, (tuple, list)):

            out = []
            for item in batch:
                if isinstance(item, _torch.Tensor):
                    item = item.to(self.device)
                out.append(item)
        return out
