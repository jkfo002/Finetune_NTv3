import os
import pyBigWig
import h5py
from pathlib import Path
from pyfaidx import Fasta
import pyarrow.dataset as ds
import polars as pl
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Iterator
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pytorch_lightning as lightning

from .utils import crop_center, transform_fn

# Process-local cache for file handles (one per worker process)
# This allows safe multi-worker DataLoader usage
_fasta_cache = {}  # Maps (process_id, file_path) -> Fasta handle
_bigwig_cache = {}  # Maps (process_id, file_path) -> pyBigWig handle
_h5_cache: dict[tuple[int, str], h5py.File] = {}


def _get_fasta_handle(fasta_path: str) -> Fasta:
    """Get or create a FASTA file handle for the current process."""
    process_id = os.getpid()
    abs_path = str(Path(fasta_path).resolve())
    cache_key = (process_id, abs_path)
    
    if cache_key not in _fasta_cache:
        _fasta_cache[cache_key] = Fasta(abs_path, as_raw=True, sequence_always_upper=True)
    
    return _fasta_cache[cache_key]

def _get_bigwig_handle(bigwig_path: str) -> pyBigWig.pyBigWig:
    """Get or create a BigWig file handle for the current process."""
    process_id = os.getpid()
    abs_path = str(Path(bigwig_path).resolve())
    cache_key = (process_id, abs_path)
    
    if cache_key not in _bigwig_cache:
        # Check if file exists before trying to open
        if not Path(abs_path).exists():
            raise FileNotFoundError(
                f"BigWig file not found: {abs_path}\n"
                f"Original path: {bigwig_path}\n"
                f"Current working directory: {os.getcwd()}"
            )
        
        try:
            _bigwig_cache[cache_key] = pyBigWig.open(abs_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open BigWig file: {abs_path} with error: {str(e)}\n"
                f"File exists: {Path(abs_path).exists()}\n"
                f"File size: {Path(abs_path).stat().st_size if Path(abs_path).exists() else 'N/A'} bytes"
            ) from e
    
    return _bigwig_cache[cache_key]

def _get_h5_handle(h5_path: str, mode: str = "r") -> h5py.File:
    """
    Get or create a process- and thread-local h5py file handle.
    
    Since h5py is NOT thread-safe, each thread must have its own File object.
    This function caches handles per (process_id, thread_id, file_path).
    
    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    mode : str
        File open mode. Default is "r" (read-only). Use "r" for training.
        
    Returns
    -------
    h5py.File
        A cached, thread-local file handle.
    """
    pid = os.getpid()
    abs_path = str(Path(h5_path).resolve())
    cache_key = (pid, abs_path)

    if cache_key not in _h5_cache:
        h5_file = Path(abs_path)
        if not h5_file.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {abs_path}\n"
                f"Original path: {h5_path}\n"
                f"Current working directory: {os.getcwd()}"
            )
        
        try:
            # Open in read-only mode by default
            _h5_cache[cache_key] = h5py.File(abs_path, mode, libver='latest')
        except Exception as e:
            file_size = h5_file.stat().st_size if h5_file.exists() else 'N/A'
            raise RuntimeError(
                f"Failed to open HDF5 file: {abs_path} (mode={mode})\n"
                f"Error: {e}\n"
                f"File exists: True\n"
                f"File size: {file_size} bytes"
            ) from e

    return _h5_cache[cache_key]

class GenomeBigWigDataset(Dataset):
    """
    A PyTorch dataset to access a reference genome and bigwig tracks. The dataset is 
    compatible with multi-worker DataLoaders (using process-local file handles and lazy 
    loading). For each sample, a random genomic region is picked from the specified split,
    and a random window of length `sequence_length` within that region is returned.
    """

    def __init__(
        self,
        fasta_path: str,
        bigwig_path_list: list[str],
        chrom_regions: pd.DataFrame,
        sequence_length: int,
        num_samples: int,
        tokenizer: AutoTokenizer,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        keep_target_center_fraction: float = 1.0,
    ):
        super().__init__()

        # Store paths instead of opening files immediately (for multi-worker compatibility)
        self.fasta_path = fasta_path
        self.bigwig_path_list = bigwig_path_list
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.transform_fn = transform_fn
        self.keep_target_center_fraction = keep_target_center_fraction
        self.chrom_regions = chrom_regions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample a random region from the valid regions
        chrom, start, end, _, _, region_start, region_end = self.chrom_regions.iloc[idx]
        # Sequence - get FASTA handle lazily (cached per worker process)
        fasta = _get_fasta_handle(self.fasta_path)
        seq = fasta[chrom][region_start - 1 : region_end - 1]  # string slice
        # Tokenize with padding and truncation to ensure consistent lengths for batching
        tokenized = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"][0]  # Shape: (max_length,)

        # Signal from bigWig tracks (numpy array) -> torch tensor
        # Get BigWig handles lazily (cached per worker process)
        bigwig_targets = np.array([
            _get_h5_handle(bw_path)[chrom][region_start - 1 : region_end - 1]
            for bw_path in self.bigwig_path_list
        ])  # shape (num_tracks, seq_len)
        # Transpose to (seq_len, num_tracks)
        bigwig_targets = bigwig_targets.T
        # Crop targets to center fraction
        if self.keep_target_center_fraction < 1.0:
            bigwig_targets = crop_center(bigwig_targets, self.keep_target_center_fraction)
        # pyBigWig returns NaN where no data; turn NaN into 0
        bigwig_targets = torch.tensor(bigwig_targets, dtype=torch.float32)
        bigwig_targets = torch.nan_to_num(bigwig_targets, nan=0.0)

        # Apply scaling to targets
        bigwig_targets = self.transform_fn(bigwig_targets)

        sample = {
            "tokens": tokens,
            "bigwig_targets": bigwig_targets,
            "chrom": chrom,
            "start": start,
            "end": end,
        }
        return sample

class GenomeBigWigDataset_my(Dataset):
    """
    A PyTorch dataset to access a reference genome and bigwig tracks. The dataset is 
    compatible with multi-worker DataLoaders (using process-local file handles and lazy 
    loading). For each sample, a random genomic region is picked from the specified split,
    and a random window of length `sequence_length` within that region is returned.

    In My version, I precompute tracks and load from disk in parquart format.
    """

    def __init__(
        self,
        fasta_path: str,
        chrom_regions: ds.dataset,
        sequence_length: int,
        num_samples: int,
        tokenizer: AutoTokenizer,
        keep_target_center_fraction: float = 1.0,
        cache_line: int = 4000,
    ):
        super().__init__()

        # Store paths instead of opening files immediately (for multi-worker compatibility)
        self.fasta_path = fasta_path
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.keep_target_center_fraction = keep_target_center_fraction
        self.chrom_regions = chrom_regions.to_table().to_pandas() # 可能爆内存

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample a random region from the valid regions
        chrom, start, end, id, _, bigwig_targets, track_shapes = self.chrom_regions.iloc[idx]

        # Sequence - get FASTA handle lazily (cached per worker process)
        fasta = _get_fasta_handle(self.fasta_path)
        seq = fasta[chrom][start:end]  # string slice
        # Tokenize with padding and truncation to ensure consistent lengths for batching
        tokenized = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"][0]  # Shape: (max_length,)

        # Signal from bigWig tracks (numpy array) -> torch tensor
        # Get BigWig handles lazily (cached per worker process)
        # bigwig_targets = np.array([
        #     _get_bigwig_handle(bw_path).values(chrom, start, end, numpy=True)
        #     for bw_path in self.bigwig_path_list
        # ])  # shape (num_tracks, seq_len)
        # # Transpose to (seq_len, num_tracks)
        # bigwig_targets = bigwig_targets.T
        # Crop targets to center fraction
        
        bigwig_targets = bigwig_targets.reshape(track_shapes)
        bigwig_targets = torch.tensor(bigwig_targets, dtype=torch.float32)
        bigwig_targets = transform_fn(bigwig_targets)
        if self.keep_target_center_fraction < 1.0:
            bigwig_targets = crop_center(bigwig_targets, self.keep_target_center_fraction)
        # pyBigWig returns NaN where no data; turn NaN into 0
        
        #bigwig_targets = torch.nan_to_num(bigwig_targets, nan=0.0)

        ## Apply scaling to targets
        #bigwig_targets = self.transform_fn(bigwig_targets)

        sample = {
            "tokens": tokens,
            "bigwig_targets": bigwig_targets,
            "chrom": chrom,
            "start": start,
            "end": end,
        }
        return sample

class GenomeBigWigDataset_myIterable(IterableDataset):
    """
    A PyTorch dataset using Polars for lazy, streaming Parquet loading.
    Replaces pyarrow.dataset with Polars LazyFrame + streaming.
    """

    def __init__(
        self,
        fasta_path: str,
        parquet_path: list[str],  # 改为 Parquet 路径（支持通配符如 "data/*.parquet"）
        sequence_length: int,
        tokenizer,
        keep_target_center_fraction: float = 1.0,
        batch_size: int = 1,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.parquet_path = parquet_path
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.keep_target_center_fraction = keep_target_center_fraction
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # 构建 Polars LazyFrame（惰性，不加载数据）
        lf = pl.scan_parquet(self.parquet_path, missing_columns="insert")
        self.num_samples = lf.select(pl.len()).collect().item()

        # 计算每个 worker 负责的样本范围
        samples_per_worker = self.num_samples // num_workers
        start_sample = worker_id * samples_per_worker
        end_sample = start_sample + samples_per_worker
        if worker_id == num_workers - 1:
            end_sample = self.num_samples  # 最后一个 worker 吃掉余数

        # 【关键】切片：跳过前面的样本，只取本 worker 的部分
        # 注意：Polars 的 slice 是 (offset, length)
        lf_worker = lf.slice(offset=start_sample, length=end_sample - start_sample)

        # 【流式迭代】按 batch_size 切片处理
        # 使用 iter_slices 生成每批 DataFrame（急切，但小批量）
        for df_batch in lf_worker.collect(streaming=True).iter_slices(self.batch_size):
            yield self._convert_batch_to_dataset(df_batch)

    def _stack_bigwig_targets(self, bigwig_targets_list, track_shapes_list):
        output_list = []
        for b, s in zip(bigwig_targets_list, track_shapes_list):
            # 假设 b 是 bytes 或 list，需反序列化（根据你写入方式调整）
            if isinstance(b, bytes):
                arr = np.frombuffer(b, dtype=np.float32)
            else:
                arr = np.array(b, dtype=np.float32)  # 如果是 list
            arr = arr.reshape(s)
            tensor = torch.tensor(arr, dtype=torch.float32)
            if self.keep_target_center_fraction < 1.0:
                tensor = crop_center(tensor, self.keep_target_center_fraction)
            output_list.append(tensor)
        return torch.stack(output_list, dim=0)

    def _convert_batch_to_dataset(self, df_batch: pl.DataFrame):
        # 转为 Python 对象（避免 Arrow 类型问题）
        chrom_list = df_batch["chrom"].to_list()
        start_list = df_batch["start"].to_list()
        end_list = df_batch["end"].to_list()
        bigwig_targets_list = df_batch["tracks"].to_list()          # 假设是 bytes 或 list
        track_shapes_list = df_batch["track_shapes"].to_list()      # 假设是 tuple 或 list

        seq_list = []
        for chrom, start, end in zip(chrom_list, start_list, end_list):
            seq = _get_fasta_handle(self.fasta_path)[chrom][start:end]
            seq_list.append(seq)

        # Tokenize all sequences in batch
        tokenized = self.tokenizer(
            seq_list,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]  # Shape: (batch_size, max_length)

        bigwig_targets = self._stack_bigwig_targets(bigwig_targets_list, track_shapes_list)

        return {
            "tokens": input_ids,
            "bigwig_targets": bigwig_targets
        }

from .decorator import NUC_CONFIG
class GenomeBigWigDataset_Nucl_Depend(Dataset):
    """
    A PyTorch dataset to access a reference genome and bigwig tracks. The dataset is 
    compatible with multi-worker DataLoaders (using process-local file handles and lazy 
    loading). For each sample, a random genomic region is picked from the specified split,
    and a random window of length `sequence_length` within that region is returned.

    init only with a region as a dict, like {'chrom': 'chr1', 'start': 0, 'end': 1000},
    and it will mutate all pos in a dict and return mutated seqs, then send to tokenizer.
    """

    def __init__(
        self,
        fasta_path: str,
        chrom_regions: Dict,
        sequence_length: int,
        tokenizer: AutoTokenizer,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()

        # Store paths instead of opening files immediately (for multi-worker compatibility)
        self.fasta_path = fasta_path
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.transform_fn = transform_fn
        
        # mutate all pos in a dict and return mutated seqs
        chrom, start, end = chrom_regions['chrom'], chrom_regions['start'], chrom_regions['end']

        seq = _get_fasta_handle(self.fasta_path)[chrom][start:end].upper()
        mutated_sequences = {'seq':[], 'mutation_pos':[], 'nuc':[], 'var_nt_idx':[]}
        mutated_sequences['seq'].append(seq)
        mutated_sequences['mutation_pos'].append(-1)
        mutated_sequences['nuc'].append('real_sequence')
        mutated_sequences['var_nt_idx'].append(-1)

        mutate_until_position = len(seq)
        for i in range(mutate_until_position):
            for nuc in ['A', 'C', 'G', 'T']:
                if nuc != seq[i]:
                    mutated_sequences['seq'].append(seq[:i] + nuc + seq[i+1:])
                    mutated_sequences['mutation_pos'].append(i)
                    mutated_sequences['nuc'].append(nuc)
                    mutated_sequences['var_nt_idx'].append(NUC_CONFIG.NUC_TAB[nuc])

        self.mutations_df = pd.DataFrame(mutated_sequences)

    def __len__(self):
        return len(self.mutations_df)

    def __getitem__(self, idx):
        # Sample a random region from the valid regions
        seq, mutation_pos, nuc, var_nt_idx = self.mutations_df.iloc[idx]
        # Sequence - get FASTA handle lazily (cached per worker process)
        # Tokenize with padding and truncation to ensure consistent lengths for batching
        tokenized = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"][0]  # Shape: (max_length,)

        # Signal from bigWig tracks (numpy array) -> torch tensor
        # Get BigWig handles lazily (cached per worker process)

        sample = {
            "tokens": tokens,
            "mutation_pos": mutation_pos,
            "nuc": nuc,
            "var_nt_idx": var_nt_idx,
        }

        return sample