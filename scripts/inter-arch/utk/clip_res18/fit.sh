#!/bin/bash


### fitnet stage 1
# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s1 \
--pretrained-teacher runs/utk/baseline/42/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s1 \
--seed 42 \
--log-dir clip_res18 &

# seed=0
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s1 \
--root runs/inter-arch/utk/fit-s1 \
--pretrained-teacher runs/utk/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_res18 &

# seed=123
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset utk --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s1 \
--pretrained-teacher runs/utk/baseline/123/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s1 \
--seed 123 \
--log-dir clip_res18 &

# seed=1
CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset utk --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s1 \
--pretrained-teacher runs/utk/baseline/1/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s1 \
--seed 1 \
--log-dir clip_res18 &

# seed=2
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset utk --epochs 10 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s1 \
--pretrained-teacher runs/utk/baseline/2/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s1 \
--seed 2 \
--log-dir clip_res18 &

wait


# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk --epochs 40 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s2 --lr 0.0001 \
--pretrained-teacher runs/utk/baseline/42/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s2 \
--continue-student runs/inter-arch/utk/fit-s1/42/clip_res18/checkpoints/latest.pth \
--seed 42 \
--log-dir clip_res18 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset utk --epochs 40 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s2 --lr 0.0001 \
--root runs/inter-arch/utk/fit-s2 \
--continue-student runs/inter-arch/utk/fit-s1/0/clip_res18/checkpoints/latest.pth \
--pretrained-teacher runs/utk/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_res18 &

# seed=123
CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset utk --epochs 40 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s2 --lr 0.0001 \
--pretrained-teacher runs/utk/baseline/123/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s2 \
--continue-student runs/inter-arch/utk/fit-s1/123/clip_res18/checkpoints/latest.pth \
--seed 123 \
--log-dir clip_res18 &

# seed=1
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset utk --epochs 40 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s2 --lr 0.0001 \
--pretrained-teacher runs/utk/baseline/1/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s2 \
--continue-student runs/inter-arch/utk/fit-s1/1/clip_res18/checkpoints/latest.pth \
--seed 1 \
--log-dir clip_res18 &

# seed=2
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset utk --epochs 40 \
--student resnet18 \
--teacher clip --vlm clip \
--fitnet-s2 --lr 0.0001 \
--pretrained-teacher runs/utk/baseline/2/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fit-s2 \
--continue-student runs/inter-arch/utk/fit-s1/2/clip_res18/checkpoints/latest.pth --seed 2 \
--log-dir clip_res18 &

wait
