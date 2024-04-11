# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava \
--distill --vlm flava \
--pretrained-teacher runs/celeba/baseline/42/flava/checkpoints/latest.pth \
--root runs/celeba/kd --epochs 10 \
--seed 42 \
--log-dir flava_flava &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava \
--distill --vlm flava \
--root runs/celeba/kd --epochs 10 \
--pretrained-teacher runs/celeba/baseline/0/flava/checkpoints/latest.pth \
--seed 0 \
--log-dir flava_flava &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava \
--distill --vlm flava \
--root runs/celeba/kd --epochs 10 \
--pretrained-teacher runs/celeba/baseline/1/flava/checkpoints/latest.pth \
--seed 1 \
--log-dir flava_flava &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava \
--distill --vlm flava \
--root runs/celeba/kd --epochs 10 \
--pretrained-teacher runs/celeba/baseline/2/flava/checkpoints/latest.pth \
--seed 2 \
--log-dir flava_flava &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student flava \
--teacher flava \
--distill --vlm flava \
--pretrained-teacher runs/celeba/baseline/123/flava/checkpoints/latest.pth \
--root runs/celeba/kd --epochs 10 \
--seed 123 \
--log-dir flava_flava &
