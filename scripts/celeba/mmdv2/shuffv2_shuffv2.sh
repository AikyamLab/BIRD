# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/42/shuffv2/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 42 \
--log-dir shuffv2_shuffv2 &

# seed=0
CUDA_VISIBLE_DEVICES=1 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--loss mmd \
--root runs/celeba/mmdv2 --mmdv2 \
--pretrained-teacher runs/celeba/baseline/0/shuffv2/checkpoints/latest.pth \
--seed 0 \
--log-dir shuffv2_shuffv2 &

# seed=123
CUDA_VISIBLE_DEVICES=2 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/123/shuffv2/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 123 \
--log-dir shuffv2_shuffv2 &

# seed=1
CUDA_VISIBLE_DEVICES=3 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/1/shuffv2/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 1 \
--log-dir shuffv2_shuffv2 &

# seed=2
CUDA_VISIBLE_DEVICES=4 python3 main.py --lambda_ 7 --num-task-classes 2 --dataset celeba --epochs 10 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--loss mmd \
--pretrained-teacher runs/celeba/baseline/2/shuffv2/checkpoints/latest.pth \
--root runs/celeba/mmdv2 --mmdv2 \
--seed 2 \
--log-dir shuffv2_shuffv2 &