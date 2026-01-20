# modisco report -i /vepfs-C/vepfs_public/daijc/lncRNA/results/grads/modisco_all.h5 -o report/all -s report/all
# modisco meme -i /vepfs-C/vepfs_public/daijc/lncRNA/results/grads/modisco_all.h5 -o report/all/motif.meme -t PFM
tomtom -oc tomtom_out -thresh 0.1 -dist pearson -norc modisco_motifs.meme ./data/Stubersum.meme