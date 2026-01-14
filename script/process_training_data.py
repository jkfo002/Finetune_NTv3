import pandas as pd
import numpy as np
import os
import tqdm
import argparse
import pyfaidx
import h5py

# prepare data
def gene_filter(gene_df, faidx, TSS_region_len_up, TSS_region_len_down):
    """
    过滤基因数据框，保留在基因组中的基因
    :param gene_df: 基因数据框
    :param faidx: pyfaidx
    :return: 过滤后的基因数据框
    """
    # 过滤基因数据框，保留在基因组中的基因
    input_gene_sets = []
    for _, row in gene_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        gene_id = row["id"]
        if start - TSS_region_len_up > 0 and start + TSS_region_len_down < len(faidx[chrom]):
            input_gene_sets.append(gene_id)
    filted_gene_df = gene_df[gene_df["id"].isin(input_gene_sets)]

    return filted_gene_df

def _h5ad(chrom, TSS, TSS_region_len_up, TSS_region_len_down, h5ad_tracks):

    track_embeddings = np.zeros((TSS_region_len_up + TSS_region_len_down, len(h5ad_tracks)), dtype=np.float32)

    for i, h5ad_track in enumerate(h5ad_tracks):
        with h5py.File(h5ad_track, "r") as f:
            track_embeddings[:, i] = f[chrom][TSS-TSS_region_len_up:TSS+TSS_region_len_down]

    track_embeddings_shape = track_embeddings.shape
    track_embeddings = track_embeddings.flatten()

    return track_embeddings, track_embeddings_shape

def load_Data(gene_df, h5ad_tracks, faidx, TSS_region_len_up, TSS_region_len_down):
    """
    加载基因数据和h5ad文件
    :param gene_df: 基因数据框
    :param h5ad_tracks: h5ad文件列表
    :param TSS_region_len_up: TSS区域长度 up
    :param TSS_region_len_down: TSS区域长度 down
    """

    filted_gene_df = gene_filter(gene_df, faidx, TSS_region_len_up, TSS_region_len_down)

    tracks = []
    track_embeddings_shapes = []

    for _, row in tqdm.tqdm(filted_gene_df.iterrows(), total=filted_gene_df.shape[0]):
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        gene_id = row["id"]

        TSS = start
        if len(h5ad_tracks) > 0:
            track_embeddings, track_embeddings_shape = _h5ad(chrom, TSS, TSS_region_len_up, TSS_region_len_down, h5ad_tracks)
            tracks.append(track_embeddings)
            track_embeddings_shapes.append(track_embeddings_shape)
        else:
            pass
    
    if len(tracks) > 0:
        filted_gene_df["tracks"] = [t.tolist() for t in tracks]
        filted_gene_df["track_shapes"] = [list(s) for s in track_embeddings_shapes]

    return filted_gene_df

# parse args
parser = argparse.ArgumentParser(description="Prepare training data")
parser.add_argument("-i", "--gene_bed", type=str, required=True, help="Gene bed file")
parser.add_argument("-g","--genome_fa", type=str, required=True, help="Genome fasta file")
parser.add_argument("-c","--chrom", type=str, required=True, help="Process chr")
parser.add_argument("-tu", "--TSS_region_len_up", type=int, default=12800, help="TSS region up length")
parser.add_argument("-td", "--TSS_region_len_down", type=int, default=12800, help="TSS region down length")
parser.add_argument("--track_df", type=str, required=True, help="Track bigwig dataframe")
parser.add_argument("--output", type=str, required=True, help="Output parquet file")
args = parser.parse_args()

# train config
TSS_region_len = args.TSS_region_len_up + args.TSS_region_len_down
TSS_region_len_up = args.TSS_region_len_up
TSS_region_len_down = args.TSS_region_len_down

# load data
gene_df = pd.read_csv(args.gene_bed, header=None, index_col=None, sep="\t", names=["chrom", "start", "end", "id", 'lncRNA'])
gene_df = gene_df[gene_df['lncRNA'] == 'mRNA']
gene_df = gene_df[gene_df['chrom'] == args.chrom]

# load genome
genome = pyfaidx.Fasta(args.genome_fa)
# load tracks
track_data_df = pd.read_csv(args.track_df, header=None, index_col=None, sep="\t", names=["type", "id", "bw_fname", "h5_fname"])

# ordered tracks
track_bw_tracks_list = track_data_df["h5_fname"].tolist()

filter_gene_df = load_Data(gene_df, track_bw_tracks_list, genome, TSS_region_len_up, TSS_region_len_down)
filter_gene_df.to_parquet(args.output)

