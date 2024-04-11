# seed=42
CUDA_VISIBLE_DEVICES=5 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/42/clip/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--seed 42 \
--log-dir clip_clip &

# seed=0
CUDA_VISIBLE_DEVICES=5 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--pretrained-teacher runs/celeba/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_clip &

CUDA_VISIBLE_DEVICES=6 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--pretrained-teacher runs/celeba/baseline/1/clip/checkpoints/latest.pth \
--seed 1 \
--log-dir clip_clip &

CUDA_VISIBLE_DEVICES=7 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--pretrained-teacher runs/celeba/baseline/2/clip/checkpoints/latest.pth \
--seed 2 \
--log-dir clip_clip &

# seed=123
CUDA_VISIBLE_DEVICES=4 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student clip \
--teacher clip --vlm clip \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/123/clip/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 --epochs 10 \
--seed 123 \
--log-dir clip_clip &