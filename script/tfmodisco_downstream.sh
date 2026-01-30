# modisco report -i /vepfs-C/vepfs_public/daijc/lncRNA/results/grads/modisco_all.h5 -o report/all -s report/all
# modisco meme -i /vepfs-C/vepfs_public/daijc/lncRNA/results/grads/modisco_all.h5 -o report/all/motif.meme -t PFM
modisco seqlet-bed -i /vepfs-C/vepfs_public/daijc/lncRNA/results/grads/modisco_ordered_all_win25600.h5 \
    -p /vepfs-C/vepfs_public/daijc/lncRNA/training_data/merged_combined_sorted_mRNA.bed \
    -o /vepfs-C/vepfs_public/daijc/lncRNA/results/grads/report/all_ordered_win25600/seqlet.bed
tomtom -oc tomtom_out -thresh 0.1 -dist pearson -norc modisco_motifs.meme ./data/Stubersum.meme