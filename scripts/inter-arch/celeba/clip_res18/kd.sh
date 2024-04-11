#!/bin/bash

# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--distill \
--pretrained-teacher runs/celeba/baseline/42/clip/checkpoints/latest.pth \
--root runs/inter-arch/celeba/clip_res18/kd \
--seed 42 \
--log-dir clip_res18 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--distill \
--root runs/inter-arch/celeba/clip_res18/kd \
--pretrained-teacher runs/celeba/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_res18 &

# seed=123
CUDA_VISIBLE_DEVICES=2 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--distill \
--pretrained-teacher runs/celeba/baseline/123/clip/checkpoints/latest.pth \
--root runs/inter-arch/celeba/clip_res18/kd \
--seed 123 \
--log-dir clip_res18 &

# seed=1
CUDA_VISIBLE_DEVICES=3 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--distill \
--pretrained-teacher runs/celeba/baseline/1/clip/checkpoints/latest.pth \
--root runs/inter-arch/celeba/clip_res18/kd \
--seed 1 \
--log-dir clip_res18 &

# seed=2
CUDA_VISIBLE_DEVICES=4 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--distill \
--pretrained-teacher runs/celeba/baseline/2/clip/checkpoints/latest.pth \
--root runs/inter-arch/celeba/clip_res18/kd --seed 2 \
--log-dir clip_res18 &

wait
