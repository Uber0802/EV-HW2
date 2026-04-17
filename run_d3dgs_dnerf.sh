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
  # --enable_speede_tricks \
  # --speede_prune_from_iter 7000 \
  # --speede_prune_interval 4000 \
  # --gflow_flag \
  # --gflow_iteration 15000 \
  # --gflow_num 2048



python render.py \
  -m output/${exp_name} ${blender_arg} -s ${data_path} --eval
# if gflow triggered
# python render.py \
#   -m output/${exp_name} ${blender_arg} -s ${data_path} --eval --gflow_flag

python metrics.py \
  -m output/${exp_name}

