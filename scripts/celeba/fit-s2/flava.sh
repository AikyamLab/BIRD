# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/flava_flava/checkpoints/latest.pth \
--student flava \
--teacher flava \
--fitnet-s2 --vlm flava \
--pretrained-teacher runs/celeba/baseline/42/flava/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/42/flava_flava/checkpoints/latest.pth \
--seed 42 \
--log-dir flava_flava &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/flava_flava/checkpoints/latest.pth \
--student flava \
--teacher flava \
--fitnet-s2 --vlm flava \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/flava_flava/checkpoints/latest.pth \
--pretrained-teacher runs/celeba/baseline/0/flava/checkpoints/latest.pth \
--seed 0 \
--log-dir flava_flava &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/flava_flava/checkpoints/latest.pth \
--student flava \
--teacher flava \
--fitnet-s2 --vlm flava \
--pretrained-teacher runs/celeba/baseline/123/flava/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/123/flava_flava/checkpoints/latest.pth \
--seed 123 \
--log-dir flava_flava &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/flava_flava/checkpoints/latest.pth \
--student flava \
--teacher flava \
--fitnet-s2 --vlm flava \
--pretrained-teacher runs/celeba/baseline/1/flava/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/1/flava_flava/checkpoints/latest.pth \
--seed 1 \
--log-dir flava_flava &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/flava_flava/checkpoints/latest.pth \
--student flava \
--teacher flava \
--fitnet-s2 --vlm flava \
--pretrained-teacher runs/celeba/baseline/2/flava/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/2/flava_flava/checkpoints/latest.pth \
--seed 2 \
--log-dir flava_flava &