#!/bin/bash


# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk --epochs 10 \
--base clip --vlm clip \
--root runs/utk/baseline_overfit \
--continue-student runs/utk/baseline/42/clip/checkpoints/latest.pth \
--seed 42 \
--log-dir clip &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset utk --epochs 10 \
--base clip --vlm clip \
--root runs/utk/baseline_overfit \
--continue-student runs/utk/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip &

# seed=123
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk --epochs 10 \
--base clip --vlm clip \
--root runs/utk/baseline_overfit \
--continue-student runs/utk/baseline/123/clip/checkpoints/latest.pth \
--seed 123 \
--log-dir clip &

# seed=1
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset utk --epochs 10 \
--base clip --vlm clip \
--root runs/utk/baseline_overfit \
--continue-student runs/utk/baseline/1/clip/checkpoints/latest.pth \
--seed 1 \
--log-dir clip &

# seed=2
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk --epochs 10 \
--base clip --vlm clip \
--root runs/utk/baseline_overfit \
--continue-student runs/utk/baseline/2/clip/checkpoints/latest.pth --seed 2 \
--log-dir clip &

wait


# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk \
--student resnet18 \
--teacher clip \
--temperature 100 \
--fwd-loss --vlm clip \
--pretrained-teacher runs/utk/baseline_overfit/42/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fwd_t100/ \
--seed 42 \
--log-dir clip_res18 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset utk \
--student resnet18 \
--teacher clip \
--temperature 100 \
--fwd-loss --vlm clip \
--root runs/inter-arch/utk/fwd_t100/ \
--pretrained-teacher runs/utk/baseline_overfit/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_res18 &

# seed=1
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk \
--student resnet18 \
--teacher clip \
--temperature 100 \
--fwd-loss --vlm clip \
--root runs/inter-arch/utk/fwd_t100/ \
--pretrained-teacher runs/utk/baseline_overfit/1/clip/checkpoints/latest.pth \
--seed 1 \
--log-dir clip_res18 &

# seed=2
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset utk \
--student resnet18 \
--teacher clip \
--temperature 100 \
--fwd-loss --vlm clip \
--root runs/inter-arch/utk/fwd_t100/ \
--pretrained-teacher runs/utk/baseline_overfit/2/clip/checkpoints/latest.pth \
--seed 2 \
--log-dir clip_res18 &

# seed=123
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset utk \
--student resnet18 \
--teacher clip \
--temperature 100 \
--fwd-loss --vlm clip \
--pretrained-teacher runs/utk/baseline_overfit/123/clip/checkpoints/latest.pth \
--root runs/inter-arch/utk/fwd_t100/ --seed 123 \
--log-dir clip_res18 &

wait
