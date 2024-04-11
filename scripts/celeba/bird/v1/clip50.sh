# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip50 \
--teacher clip50 --vlm clip50 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/42/clip50/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 5e-3 \
--seed 42 \
--log-dir clip50_clip50 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip50 \
--teacher clip50 --vlm clip50 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/0/clip50/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 5e-3 \
--seed 0 \
--log-dir clip50_clip50 &

# seed=1
CUDA_VISIBLE_DEVICES=2 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip50 \
--teacher clip50 --vlm clip50 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/1/clip50/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 5e-3 \
--seed 1 \
--log-dir clip50_clip50 &

# seed=2
CUDA_VISIBLE_DEVICES=3 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip50 \
--teacher clip50 --vlm clip50 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/2/clip50/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 5e-3 \
--seed 2 \
--log-dir clip50_clip50 &


# seed=123
CUDA_VISIBLE_DEVICES=4 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip50 \
--teacher clip50 --vlm clip50 \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/123/clip50/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 5e-3 \
--seed 123 \
--log-dir clip50_clip50 &