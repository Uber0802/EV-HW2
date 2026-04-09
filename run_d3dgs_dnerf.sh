#!/bin/bash

# Deformable 3DGS
#exp_name="deformable_vrig_3dprinter_prune_2"

exp_name="deformable_bouncing"


data_path="data/dnerf/data/bouncingballs"

# if data_path is a dnerf dataset, use is_blender=True
# if data_path is a hypernerf dataset, use is_blender=False
if [ -f "${data_path}/transforms_train.json" ]; then
  blender_arg='--is_blender'      # D-NeRF / Blender format
elif [ -f "${data_path}/metadata.json" ]; then
  blender_arg=''                  # HyperNeRF / Nerfies format
else
  echo "Unknown dataset format: ${data_path}"
  exit 1
fi

python train.py \
  -s ${data_path} \
  --method deformable \
  ${blender_arg} --eval \
  -m output/${exp_name} \
#   --enable_speede_tricks \
#   --speede_prune_from_iter 7000 \
#   --speede_prune_interval 4000 \
#   --speede_densify_prune_ratio 0.30 \
#   --speede_after_densify_prune_ratio 0.15 \
#   --speede_use_tss --speede_use_vc \
#   --gflow_flag \
#   --gflow_iteration 15000 \
#   --gflow_num 2048 \
#   --gflow_opt 2 \





python render.py \
  -m output/${exp_name} ${blender_arg} -s ${data_path} --eval
# if gflow triggered
# python render.py \
#   -m output/${exp_name} ${blender_arg} -s ${data_path} --eval --gflow_flag
python metrics.py \
  -m output/${exp_name}



# python train.py --method 4dgs \
#   --enable_4dgs_temporal_track \
#   --lambda_4d_speed_smooth 0.01 \
#   --lambda_4d_cov_smooth 0.005 \
#   --lambda_4d_neighbor_speed 0.01 \
#   --temporal_dt_frames 1

# python train.py --method deformable \
#   --enable_hybrid_rigid_track \
#   --lambda_hybrid_rigid_residual 0.01

# 4DGS

# CUDA_VISIBLE_DEVICES=1 /mnt/home/uber/miniconda3/envs/3dgs/bin/python train.py -s data/dnerf/bouncingballs --method 4dgs --is_blender --eval --white_background -m output/4dgs_bouncing


# CUDA_VISIBLE_DEVICES=1 /mnt/home/uber/miniconda3/envs/3dgs/bin/python render.py -m output/4dgs_bouncing --is_blender -s data/dnerf/bouncingballs --eval

# CUDA_VISIBLE_DEVICES=1 /mnt/home/uber/miniconda3/envs/3dgs/bin/python metrics.py -m output/4dgs_bouncing

