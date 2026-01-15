import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def visualization_channels(targets, preds, save_path=None, channels=None):
    """
    targets: (1, len, channels)
    preds: (1, len, channels)
    """
    assert targets.shape == preds.shape, "targets and preds must have the same shape"
    assert targets.shape[0] == 1, "targets and preds must have batch size of 1"
    assert preds.shape[0] == 1, "targets and preds must have batch size of 1"
    
    if channels is None:

        plt.figure(figsize=(15, 4 * targets.shape[-1]))        
        for i in range(targets.shape[-1]):

            plot_target = targets.squeeze(0)[:, i] # (len)
            # plot_target = normaliztion(plot_target) # z-score the targe
            plot_pred = preds.squeeze(0)[:, i] # (len)
            # plot_pred = normaliztion(plot_pred) # z-score the pred  

            plt.subplot(targets.shape[-1]*2, 1, i*2+1)
            plt.plot(plot_target, color="#779a92")
            plt.gca().text(0.05, 0.9, f'Target channel {i+1}',
                        transform=plt.gca().transAxes,
                        fontsize=10, color='#9aadbe')
            plt.subplot(targets.shape[-1]*2, 1, i*2+2)
            plt.plot(plot_pred, color="#9aadbe")
            plt.gca().text(0.05, 0.9, f'Pred channel {i+1}',
                        transform=plt.gca().transAxes,
                        fontsize=10, color='#9aadbe')
    else:

        plt.figure(figsize=(15, 4 * len(channels)))        
        for i in range(len(channels)):

            plot_target = targets.squeeze(0)[:, channels[i]] # (len)
            # plot_target = normaliztion(plot_target) # z-score the targe
            plot_pred = preds.squeeze(0)[:, channels[i]] # (len)
            # plot_pred = normaliztion(plot_pred) # z-score the pred

            plt.subplot(len(channels)*2, 1, i*2+1)
            plt.plot(plot_target, color="#779a92")
            plt.gca().text(0.05, 0.9, f'Target channel {channels + 1}',
                        transform=plt.gca().transAxes,
                        fontsize=10, color='black')
            plt.subplot(len(channels)*2, 1, i*2+2)

            plt.plot(plot_pred, color="#9aadbe")
            plt.gca().text(0.05, 0.9, f'Pred channel {channels + 1}',
                        transform=plt.gca().transAxes,
                        fontsize=10, color='black')

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

def visualization_channels_means(
    targets:np.ndarray, preds:np.ndarray, mean_order:dict, save_path=None, channels=None
):
    """
    means the replication within both target and pred channels
    targets: (1, len, channels)
    preds: (1, len, channels)
    """
    assert targets.shape == preds.shape, "targets and preds must have the same shape"
    assert targets.shape[0] == 1, "targets and preds must have batch size of 1"
    assert preds.shape[0] == 1, "targets and preds must have batch size of 1"
    _, seq_len, num_all_channels = targets.shape

    mean_channels = list(mean_order.keys())
    mean_target = np.zeros((1, seq_len, len(mean_channels)))
    mean_pred = np.zeros((1, seq_len, len(mean_channels)))

    for i in range(len(mean_channels)):

        mc = mean_channels[i]
        select_channel = mean_order[mc]
        mean_target[:, :, i] = targets[:, :, select_channel].mean(axis=-1)
        mean_pred[:, :, i] = preds[:, :, select_channel].mean(axis=-1)

    visualization_channels(mean_target, mean_pred, save_path, channels)

