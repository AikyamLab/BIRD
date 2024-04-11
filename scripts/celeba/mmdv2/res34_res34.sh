# seed=42
CUDA_VISIBLE_DEVICES=5 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/42/res34/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 42 \
--log-dir res34_res34 &

# seed=0
CUDA_VISIBLE_DEVICES=6 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 \
--pretrained-teacher runs/celeba/baseline/0/res34/checkpoints/latest.pth \
--seed 0 \
--log-dir res34_res34 &

# seed=123
CUDA_VISIBLE_DEVICES=7 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/123/res34/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 123 \
--log-dir res34_res34 &

# seed=1
CUDA_VISIBLE_DEVICES=5 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/1/res34/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 1 \
--log-dir res34_res34 &

# seed=2
CUDA_VISIBLE_DEVICES=6 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/2/res34/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 2 \
--log-dir res34_res34 &