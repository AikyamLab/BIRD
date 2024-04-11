# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher resnet18 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/42/res18/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 42 \
--log-dir res18_res18 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher resnet18 \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 \
--pretrained-teacher runs/celeba/baseline/0/res18/checkpoints/latest.pth \
--seed 0 \
--log-dir res18_res18 &

CUDA_VISIBLE_DEVICES=2 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher resnet18 \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 \
--pretrained-teacher runs/celeba/baseline/1/res18/checkpoints/latest.pth \
--seed 1 \
--log-dir res18_res18 &

CUDA_VISIBLE_DEVICES=3 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher resnet18 \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 \
--pretrained-teacher runs/celeba/baseline/2/res18/checkpoints/latest.pth \
--seed 2 \
--log-dir res18_res18 &

# seed=123
CUDA_VISIBLE_DEVICES=4 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet18 \
--teacher resnet18 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/123/res18/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 123 \
--log-dir res18_res18 &