# seed=42
CUDA_VISIBLE_DEVICES=6 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/42/flava/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--seed 42 \
--log-dir flava_flava &

# seed=0
CUDA_VISIBLE_DEVICES=6 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--pretrained-teacher runs/celeba/baseline/0/flava/checkpoints/latest.pth \
--seed 0 \
--log-dir flava_flava &

CUDA_VISIBLE_DEVICES=7 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--pretrained-teacher runs/celeba/baseline/1/flava/checkpoints/latest.pth \
--seed 1 \
--log-dir flava_flava &

CUDA_VISIBLE_DEVICES=7 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--pretrained-teacher runs/celeba/baseline/2/flava/checkpoints/latest.pth \
--seed 2 \
--log-dir flava_flava &

# seed=123
CUDA_VISIBLE_DEVICES=0 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava --vlm flava \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/123/flava/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--seed 123 \
--log-dir flava_flava &