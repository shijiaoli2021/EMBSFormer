import torch
import numpy as np


def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    if torch.linalg.norm(y, "fro") == 0:
        return torch.tensor(1, device=y.device).to(torch.float32)
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
    
def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.ne(labels, null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)

def masked_rmse_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))

def masked_mse_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        # mask = ~tf.is_nan(labels)
        mask = ~torch.isnan(labels)
    else:
        # mask = tf.not_equal(labels, null_val)
        mask = torch.ne(labels, null_val)
    # mask = tf.cast(mask, tf.float32)
    mask = mask.to(torch.float32)
    # mask /= tf.reduce_mean(mask)
    mask /= torch.mean(mask)
    # mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = tf.square(tf.subtract(preds, labels))
    loss = torch.pow(preds - labels, 2)
    loss = loss * mask
    # loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.ne(labels, null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
