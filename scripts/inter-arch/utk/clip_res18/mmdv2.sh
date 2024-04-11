#!/bin/bash



# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --lambda_ 3 --dataset utk \
--student resnet18 \
--teacher clip --vlm clip \
--loss mmd \
--pretrained-teacher runs/utk/baseline/42/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/clip_res18/mmdv2 --mmdv2 \
--seed 42 \
--log-dir clip_res18 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --lambda_ 3 --dataset utk \
--student resnet18 \
--teacher clip --vlm clip \
--loss mmd \
--root runs/inter-arch/utk/clip_res18/mmdv2 --mmdv2 \
--pretrained-teacher runs/utk/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_res18 &

CUDA_VISIBLE_DEVICES=2 python3 main.py --lambda_ 3 --dataset utk \
--student resnet18 \
--teacher clip --vlm clip \
--loss mmd \
--root runs/inter-arch/utk/clip_res18/mmdv2 --mmdv2 \
--pretrained-teacher runs/utk/baseline/1/clip/checkpoints/latest.pth \
--seed 1 \
--log-dir clip_res18 &

CUDA_VISIBLE_DEVICES=3 python3 main.py --lambda_ 3 --dataset utk \
--student resnet18 \
--teacher clip --vlm clip \
--loss mmd \
--root runs/inter-arch/utk/clip_res18/mmdv2 --mmdv2 \
--pretrained-teacher runs/utk/baseline/2/clip/checkpoints/latest.pth \
--seed 2 \
--log-dir clip_res18 &

# seed=123
CUDA_VISIBLE_DEVICES=4 python3 main.py --lambda_ 3 --dataset utk \
--student resnet18 \
--teacher clip --vlm clip \
--loss mmd \
--pretrained-teacher runs/utk/baseline/123/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/clip_res18/mmdv2 --mmdv2 \
--seed 123 \
--log-dir clip_res18 &

wait
