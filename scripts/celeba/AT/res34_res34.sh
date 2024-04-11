# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--AT \
--pretrained-teacher runs/celeba/baseline/42/res34/checkpoints/latest.pth \
--root runs/celeba/AT \
--seed 42 \
--log-dir res34_res34 &

# seed=0
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--AT \
--root runs/celeba/AT \
--pretrained-teacher runs/celeba/baseline/0/res34/checkpoints/latest.pth \
--seed 0 \
--log-dir res34_res34 &

# seed=123
CUDA_VISIBLE_DEVICES=1 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--AT \
--pretrained-teacher runs/celeba/baseline/123/res34/checkpoints/latest.pth \
--root runs/celeba/AT \
--seed 123 \
--log-dir res34_res34 &

# seed=1
CUDA_VISIBLE_DEVICES=4 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--AT \
--pretrained-teacher runs/celeba/baseline/1/res34/checkpoints/latest.pth \
--root runs/celeba/AT \
--seed 1 \
--log-dir res34_res34 &

# seed=2
CUDA_VISIBLE_DEVICES=4 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--AT \
--pretrained-teacher runs/celeba/baseline/2/res34/checkpoints/latest.pth \
--root runs/celeba/AT \
--seed 2 \
--log-dir res34_res34 &