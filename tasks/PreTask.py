import torch
import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import matplotlib.pyplot as plt
import numpy as np
import random


class PreSeqTask(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 loss="mae",
                 learning_rate: float = 1e-3,
                 weight_decay=1.5e-3,
                 data_max_val=None):
        super(PreSeqTask, self).__init__()
        self.model = model
        self._loss = loss
        self.learning_rate = learning_rate
        self.plot_test_batch_id = random.randint(0, 100)
        self.weight_decay = weight_decay
        self.data_max_val = data_max_val
        self.rmse_list = []
        self.mae_list = []
        self.loss_list = []
        self.metrics = None
        self.save_hyperparameters()
        # self.plot_batch = None
        # print("hp:",self.hparams)

    def forward(self, src):
        predictions = self.model(src)
        return predictions

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        # if self._loss == "mae":
        #     return F.l1_loss(inputs, targets)
        if self._loss == "mae":
            return F.l1_loss(inputs, targets)
        if self._loss == "mask_mae":
            return utils.metrics.masked_mae_torch(inputs, targets, 0)
        
    def training_step(self, batch, batch_idx):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #(batch_size, l, n, d_model)
        src, y = batch[:-1], batch[-1]
        if len(batch) == 2:
            src, y = batch
        pre = self(src)
        # print("target:{},pre:{}".format(y, pre))
        # if batch_idx < 5:
        #     print(pre)
        loss = self.loss(pre, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch == 0 and batch_idx == self.plot_test_batch_id:
            self.plot_batch = batch
        src, y = batch[:-1], batch[-1]
        if len(batch) == 2:
            src, y = batch
        batch_size = y.size(0)
        pre = self(src)
        loss = self.loss(pre, y)
        if self._loss == "mask_mae":
            rmse = utils.metrics.masked_rmse_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
            mae = utils.metrics.masked_mae_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
            mape = torch.tensor(utils.metrics.masked_mape_np(np.array(y.reshape(-1, 1).cpu()), np.array(pre.reshape(-1, 1).cpu()), 0))
        else:
            mape = torch.tensor(utils.metrics.masked_mape_np(np.array(y.reshape(-1, 1).cpu()), np.array(pre.reshape(-1, 1).cpu()), 0))
            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pre.reshape(batch_size, -1), y.reshape(batch_size, -1)))
            mae = torchmetrics.functional.mean_absolute_error(pre.reshape(batch_size, -1), y.reshape(batch_size, -1))
        accuracy = utils.metrics.accuracy(pre.contiguous().reshape(batch_size, -1), y.contiguous().reshape(batch_size, -1))
        r2 = utils.metrics.r2(pre.contiguous().reshape(batch_size, -1), y.contiguous().reshape(batch_size, -1))
        explained_variance = utils.metrics.explained_variance(pre.contiguous().reshape(batch_size, -1), y.contiguous().reshape(batch_size, -1))
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "MAPE": mape,
            "ExplainedVar": explained_variance,
        }
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, output):
        val_loss = torch.stack([loss["val_loss"] for loss in output]).mean()
        rmse = torch.sqrt(torch.stack([loss["RMSE"]**2 for loss in output]).mean())
        mae = torch.stack([loss["MAE"] for loss in output]).mean()
        mape = torch.stack([loss["MAPE"] for loss in output]).mean()
        accuracy = torch.stack([loss["accuracy"] for loss in output]).mean()
        r2_mean = torch.stack([loss["R2"] for loss in output]).mean()
        explainedVar_mean = torch.stack([loss["ExplainedVar"] for loss in output]).mean()
        metrics = {
            "val_loss": val_loss.item(),
            "RMSE": rmse.item(),
            "MAE": mae.item(),
            "MAPE": mape.item(),
            "accuracy": accuracy.item(),
            "R2": r2_mean.item(),
            "ExplainedVar": explainedVar_mean.item(),
        }
        self.rmse_list.append(rmse.cpu())
        self.mae_list.append(mae.cpu())
        self.loss_list.append(val_loss.cpu())
        self.log_dict(metrics)
        self.metrics = metrics
        return metrics


    def test_step(self, batch, batch_idx):
        src, y = batch[:-1], batch[-1]
        if len(batch) == 2:
            src, y = batch
        batch_size = y.size(0)
        pre = self(src)
        loss = self.loss(pre, y)
        if self._loss == "mask_mae":
            rmse = utils.metrics.masked_rmse_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
            mae = utils.metrics.masked_mae_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
            mape = torch.tensor(utils.metrics.masked_mape_np(np.array(y.reshape(-1, 1).cpu()), np.array(pre.reshape(-1, 1).cpu()), 0))
        else:
            mape = torch.tensor(utils.metrics.masked_mape_np(np.array(y.reshape(-1, 1).cpu()), np.array(pre.reshape(-1, 1).cpu()), 0))
            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pre.reshape(batch_size, -1), y.reshape(batch_size, -1)))
            mae = torchmetrics.functional.mean_absolute_error(pre.reshape(batch_size, -1), y.reshape(batch_size, -1))
        # rmse = utils.metrics.masked_rmse_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
        # mae = utils.metrics.masked_mae_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
        # mape = torch.tensor(utils.metrics.masked_mape_np(np.array(y.reshape(-1, 1).cpu()), np.array(pre.reshape(-1, 1).cpu()), 0))
        # rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pre.reshape(batch_size, -1), y.reshape(batch_size, -1)))
        # mae = utils.metrics.masked_mae_torch(pre.reshape(batch_size, -1), y.reshape(batch_size, -1), 0)
        # mape = torch.tensor(utils.metrics.masked_mape_np(np.array(y.reshape(-1, 1).cpu()), np.array(pre.reshape(-1, 1).cpu()), 0))
        accuracy = utils.metrics.accuracy(pre.contiguous().reshape(batch_size, -1), y.contiguous().reshape(batch_size, -1))
        r2 = utils.metrics.r2(pre.contiguous().reshape(batch_size, -1), y.contiguous().reshape(batch_size, -1))
        explained_variance = utils.metrics.explained_variance(pre.contiguous().reshape(batch_size, -1), y.contiguous().reshape(batch_size, -1))
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "MAPE": mape,
            "ExplainedVar": explained_variance,
        }
        self.log_dict(metrics)
        return metrics
    
    def test_epoch_end(self, output):
        val_loss = torch.stack([loss["val_loss"] for loss in output]).mean()
        rmse = torch.sqrt(torch.stack([loss["RMSE"]**2 for loss in output]).mean())
        mae = torch.stack([loss["MAE"] for loss in output]).mean()
        mape = torch.stack([loss["MAPE"] for loss in output]).mean()
        accuracy = torch.stack([loss["accuracy"] for loss in output]).mean()
        r2_mean = torch.stack([loss["R2"] for loss in output]).mean()
        explainedVar_mean = torch.stack([loss["ExplainedVar"] for loss in output]).mean()
        metrics = {
            "val_loss": val_loss.item(),
            "RMSE": rmse.item(),
            "MAE": mae.item(),
            "MAPE": mape.item(),
            "accuracy": accuracy.item(),
            "R2": r2_mean.item(),
            "ExplainedVar": explainedVar_mean.item(),
        }
        self.rmse_list.append(rmse.cpu())
        self.mae_list.append(mae.cpu())
        self.loss_list.append(val_loss.cpu())
        self.log_dict(metrics)
        self.metrics = metrics
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def plot_show(self, name, save_path):
        plt.subplot(1, 2, 1)
        plt.plot([i for i in range(1, len(self.rmse_list)+1)], self.rmse_list, color='b', linewidth=2.5, label="RMSE")
        plt.plot([i for i in range(1, len(self.mae_list)+1)], self.mae_list, color='g', linewidth=2.5, label="MAE")
        plt.title("Iterative Curve")
        plt.subplot(1, 2, 2)
        plt.plot([i for i in range(1, len(self.loss_list) + 1)], self.loss_list, color='r', linewidth=2.5, label="val_loss")
        plt.title("Loss Iterative Curve")
        img_name = name+"_"+str(min(self.rmse_list).item())+".png"
        plt.savefig(save_path+"/"+ img_name)
        return img_name

    def plot_prediction(self, name, save_path):
        if self.plot_batch is not None:
            batch = self.plot_batch
            src, y = batch[:-1], batch[-1]
            device = y.device
            self.model = self.model.to(device)
            if len(batch) == 2:
                src, y = batch
            pre = self(src)
            for i in range(0, 9):
                plt.subplot(3, 3, i+1)
                plt.plot(pre[0, i * 15, :].cpu().detach().numpy(), color='b', linewidth=2.5)
                plt.plot(y[0, i * 15, :].cpu().detach().numpy(), color='r', linewidth=2.5)
            plt.title("predict Curve pre: b, true:r")
            img_name = name + "_" + str(min(self.rmse_list).item()) + ".png"
            plt.savefig(save_path + "/" + img_name)

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser