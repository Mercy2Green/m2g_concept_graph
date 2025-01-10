export CUDA_VISIBLE_DEVICES=0

# python -m debugpy --listen 2457 --wait-for-client cg_process/sim2real/reconstruction.py
python cg_process/sim2real/reconstruction.py

