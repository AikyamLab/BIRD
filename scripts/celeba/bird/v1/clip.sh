# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip --later 5 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/42/clip/checkpoints/latest.pth \
--root runs/celeba/bird1 --epochs 10 --lr 5e-3 \
--seed 42 \
--log-dir clip_clip &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip --later 5 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/0/clip/checkpoints/latest.pth \
--root runs/celeba/bird1 --epochs 10 --lr 5e-3 \
--seed 0 \
--log-dir clip_clip &

# seed=1
CUDA_VISIBLE_DEVICES=2 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip --later 5 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/1/clip/checkpoints/latest.pth \
--root runs/celeba/bird1 --epochs 10 --lr 5e-3 \
--seed 1 \
--log-dir clip_clip &

# seed=2
CUDA_VISIBLE_DEVICES=3 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip --later 5 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/2/clip/checkpoints/latest.pth \
--root runs/celeba/bird1 --epochs 10 --lr 5e-3 \
--seed 2 \
--log-dir clip_clip &


# seed=123
CUDA_VISIBLE_DEVICES=4 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip --later 5 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/123/clip/checkpoints/latest.pth \
--root runs/celeba/bird1 --epochs 10 --lr 5e-3 \
--seed 123 \
--log-dir clip_clip &