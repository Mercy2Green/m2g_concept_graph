export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
# export NODE_RANK = 0
# export NUM_GPUS = 2
flag1="--exp_name release_r2r
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      IL.iters 800
      IL.lr 1e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 3000
      IL.load_from_ckpt False
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path /home/lg1/lujia/VLN_HGT/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
      "
#MODEL.pretrained_path /home/lg1/lujia/VLN_HGT/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
# MODEL.pretrained_path /home/lujia/VLN_HGT/VLN_HGT/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt

flag2=" --exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR /data2/vln_dataset/checkpoints/acc78_m2g/release_r2r/ckpt.iter12000.pth
      EVAL.EPISODE_COUNT -1
      IL.back_algo control
      RESULTS_DIR /data2/vln_dataset/checkpoints/eval_result/m2g_debug_100
      "
# EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag1
      # taskset -c 81-160 python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag1
      # taskset -c 37-47 python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      taskset -c 80-140 python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag2
      # taskset -c 10-36 python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag3
      ;;
esac
# CUDA_VISIBLE_DEVICES=7 bash run_r2r/main.bash train 3113
# CUDA_VISIBLE_DEVICES=7 bash run_r2r/main.bash eval 3135
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash run_r2r/main.bash eval 3135