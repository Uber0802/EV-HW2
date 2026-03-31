#!/bin/bash

# Deformable 3DGS
python train.py \
  -s data/dnerf/bouncingballs --method deformable --is_blender --eval --white_background \
  -m output/deformable_bouncing
render + metrics：


python render.py \
  -m output/deformable_bouncing --is_blender -s data/dnerf/bouncingballs --eval

python metrics.py \
  -m output/deformable_bouncing



# 4DGS

# CUDA_VISIBLE_DEVICES=1 /mnt/home/uber/miniconda3/envs/3dgs/bin/python train.py -s data/dnerf/bouncingballs --method 4dgs --is_blender --eval --white_background -m output/4dgs_bouncing


# CUDA_VISIBLE_DEVICES=1 /mnt/home/uber/miniconda3/envs/3dgs/bin/python render.py -m output/4dgs_bouncing --is_blender -s data/dnerf/bouncingballs --eval

# CUDA_VISIBLE_DEVICES=1 /mnt/home/uber/miniconda3/envs/3dgs/bin/python metrics.py -m output/4dgs_bouncing

