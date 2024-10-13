
# Set CUDA_VISIBLE_DEVICES = 7

# source activate vln_310

export CUDA_VISIBLE_DEVICES=7

python -m debugpy --listen 5678 --wait-for-client /home/lg1/lujia/VLN_HGT/m2g_concept_graph/conceptgraph/scripts/visualize_cfslam_results.py --no_clip --result_path /data0/vln_datasets/preprocessed_data/debug_finetune_online_cg/pcd_saves/full_pcd_step0_post.pkl.gz