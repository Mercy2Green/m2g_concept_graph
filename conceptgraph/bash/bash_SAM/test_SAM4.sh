export CUDA_VISIBLE_DEVICES=7

python scripts/generate_gsa_results.py \
    --dataset_root /data/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R \
    --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
    --class_set "ram" \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses
