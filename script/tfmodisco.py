
from typing import List, Literal, Union
import h5py

import argparse
import modiscolite

import numpy as np

from modiscolite.util import calculate_window_offsets

if __name__ == '__main__':

    window  = 25600 # 从中间分开进行motif discovery (center - window_size // 2, center + window_size // 2)
    max_seqlets = 20000
    size = 20
    seqlet_flank_size = 5
    trim_size = 30
    initial_flank_to_add = 10
    final_flank_to_add = 0
    n_leiden = 2
    verbose = True

    all_attributions = []
    all_sequences = []
    for i in range(3):

        attributions = np.load(f"/vepfs-C/vepfs_public/daijc/lncRNA/results/grads/gradients_ordered_{i}.npy") # (25600 x 8000, 4)
        sequences = np.load(f"/vepfs-C/vepfs_public/daijc/lncRNA/results/grads/embeddings_ordered_{i}.npy") # (25600 x 8000, 4)
        attributions = attributions.reshape(-1, 25600, 4).transpose(0, 2, 1) # (8000, 25600, 4) -> (8000, 4, 25600)
        sequences = sequences.reshape(-1, 25600, 4).transpose(0, 2, 1) # (8000, 25600, 4) -> (8000, 4, 25600)

        all_attributions.append(attributions)
        all_sequences.append(sequences)

    all_attributions = np.concatenate(all_attributions, axis=0)
    all_sequences = np.concatenate(all_sequences, axis=0)

    center = sequences.shape[2] // 2
    start, end = calculate_window_offsets(center, window)

    sequences = sequences[:, :, start:end].transpose(0, 2, 1)
    attributions = attributions[:, :, start:end].transpose(0, 2, 1)

    sequences = sequences.astype('float32')
    attributions = attributions.astype('float32')
    
    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=attributions, 
        one_hot=sequences,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=size,
        flank_size=seqlet_flank_size,
        trim_to_window_size=trim_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        target_seqlet_fdr=0.1,
        n_leiden_runs=n_leiden,
        verbose=verbose)

    modiscolite.io.save_hdf5(f"/vepfs-C/vepfs_public/daijc/lncRNA/results/grads/modisco_ordered_all_win{window}.h5", pos_patterns, neg_patterns, window)
