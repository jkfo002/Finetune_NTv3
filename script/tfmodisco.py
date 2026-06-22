
from typing import List, Literal, Union
import h5py
import argparse
import numpy as np
import os

import modiscolite
from modiscolite.util import calculate_window_offsets

def parse_args():
    args = argparse.ArgumentParser(description="TFMoDISco from onehot embedding and gradients")
    args.add_argument("--input_dir", type=str, required=True, help="input directory")
    args.add_argument("--output_dir", type=str, required=True, help="output directory")
    args.add_argument("--window", type=int, default=51200, help="window size for input")
    args.add_argument("--window_calculate", type=int, default=None, help="window size for motif discovery, replace calculate_window_offsets")
    args.add_argument("--max_seqlets", type=int, default=-1, help="max number of seqlets per metacluster, -1 means all")
    args.add_argument("--size", type=int, default=20, help="sliding window size")
    args.add_argument("--seqlet_flank_size", type=int, default=5, help="flank size for seqlets")
    args.add_argument("--trim_size", type=int, default=30, help="trim size for window")
    args.add_argument("--initial_flank_to_add", type=int, default=10, help="initial flank to add")
    args.add_argument("--final_flank_to_add", type=int, default=0, help="final flank to add")
    args.add_argument("--n_leiden", type=int, default=2, help="number of leiden clusters")
    args.add_argument("--verbose", action="store_true", help="print verbose info")
    return args.parse_args()

if __name__ == '__main__':

    args = parse_args()

    window  = args.window # 从中间分开进行motif discovery (center - window_size // 2, center + window_size // 2)
    max_seqlets = args.max_seqlets # -1 means using all seqlets changed in modiscolite.tfmodisco.TFMoDISco
    size = args.size
    seqlet_flank_size = args.seqlet_flank_size
    trim_size = args.trim_size
    initial_flank_to_add = args.initial_flank_to_add
    final_flank_to_add = args.final_flank_to_add
    n_leiden = args.n_leiden
    verbose = args.verbose
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # assert input file
    gradients_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("gradients_ordered_")]
    embeddings_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("embeddings_ordered_")]
    assert len(gradients_files) > 0, f"gradients_files: {gradients_files}, empty"
    assert len(embeddings_files) > 0, f"embeddings_files: {embeddings_files}, empty"
    assert len(gradients_files) == len(embeddings_files), f"gradients_files: {gradients_files}, embeddings_files: {embeddings_files}, not equal"

    all_attributions = []
    all_sequences = []
    for i in range(len(gradients_files)):

        gradient_file = gradients_files[i]
        embedding_file = embeddings_files[i]
        if not os.path.exists(gradient_file) or not os.path.exists(embedding_file):
            raise FileNotFoundError(f"gradient_file: {gradient_file}, embedding_file: {embedding_file}, not found")

        attributions = np.load(gradient_file) # (25600 x 8000, 4)
        sequences = np.load(embedding_file) # (25600 x 8000, 4)
        attributions = attributions.reshape(-1, window, 4).transpose(0, 2, 1) # (8000, 25600, 4) -> (8000, 4, 25600)
        sequences = sequences.reshape(-1, window, 4).transpose(0, 2, 1) # (8000, 25600, 4) -> (8000, 4, 25600)

        all_attributions.append(attributions)
        all_sequences.append(sequences)

    all_attributions = np.concatenate(all_attributions, axis=0)
    all_sequences = np.concatenate(all_sequences, axis=0)

    center = all_sequences.shape[2] // 2
    if args.window_calculate is None:
        start, end = calculate_window_offsets(center, window)
    else:
        start, end = calculate_window_offsets(center, args.window_calculate)

    all_sequences = all_sequences[:, :, start:end].transpose(0, 2, 1)
    all_attributions = all_attributions[:, :, start:end].transpose(0, 2, 1)

    all_sequences = all_sequences.astype('float32')
    all_attributions = all_attributions.astype('float32')
    
    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=all_attributions, 
        one_hot=all_sequences,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=size,
        flank_size=seqlet_flank_size,
        trim_to_window_size=trim_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        target_seqlet_fdr=0.05,
        n_leiden_runs=n_leiden,
        verbose=verbose)

    modiscolite.io.save_hdf5(os.path.join(output_dir, f"modisco_ordered_all_win{window}_n{max_seqlets}.h5"), pos_patterns, neg_patterns, window)
