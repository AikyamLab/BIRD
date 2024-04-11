# seed=42
CUDA_VISIBLE_DEVICES=6 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--AT \
--pretrained-teacher runs/celeba/baseline/42/shuffv2/checkpoints/latest.pth \
--root runs/celeba/AT \
--seed 42 \
--log-dir shuffv2_shuffv2 &

# seed=0
CUDA_VISIBLE_DEVICES=6 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--AT \
--root runs/celeba/AT \
--pretrained-teacher runs/celeba/baseline/0/shuffv2/checkpoints/latest.pth \
--seed 0 \
--log-dir shuffv2_shuffv2 &

# seed=1
CUDA_VISIBLE_DEVICES=7 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--AT \
--root runs/celeba/AT \
--pretrained-teacher runs/celeba/baseline/1/shuffv2/checkpoints/latest.pth \
--seed 1 \
--log-dir shuffv2_shuffv2 &

# seed=2
CUDA_VISIBLE_DEVICES=7 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--AT \
--root runs/celeba/AT \
--pretrained-teacher runs/celeba/baseline/2/shuffv2/checkpoints/latest.pth \
--seed 2 \
--log-dir shuffv2_shuffv2 &

# seed=123
CUDA_VISIBLE_DEVICES=7 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--AT \
--pretrained-teacher runs/celeba/baseline/123/shuffv2/checkpoints/latest.pth \
--root runs/celeba/AT \
--seed 123 \
--log-dir shuffv2_shuffv2