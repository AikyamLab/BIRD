# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/42/flava/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 1e-1 \
--seed 42 \
--log-dir flava_flava &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/0/flava/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 1e-1 \
--seed 0 \
--log-dir flava_flava &

# seed=1
CUDA_VISIBLE_DEVICES=2 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/1/flava/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 1e-1 \
--seed 1 \
--log-dir flava_flava &

# seed=2
CUDA_VISIBLE_DEVICES=3 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/2/flava/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 1e-1 \
--seed 2 \
--log-dir flava_flava &


# seed=123
CUDA_VISIBLE_DEVICES=4 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--adv \
--meta2 \
--sigmoid \
--pretrained-teacher runs/celeba/baseline/123/flava/checkpoints/latest.pth \
--root runs/celeba/bird1 --later 5 --epochs 10 --lr 1e-1 \
--seed 123 \
--log-dir flava_flava &