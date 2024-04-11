# seed=42
CUDA_VISIBLE_DEVICES=0 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss adv --adv \
--lambda_ 0.50 \
--pretrained-teacher runs/celeba/baseline/42/res34/checkpoints/latest.pth \
--root runs/celeba/adv \
--seed 42 \
--log-dir res34_res34 &

# seed=0
CUDA_VISIBLE_DEVICES=6 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss adv --adv \
--lambda_ 0.50 \
--root runs/celeba/adv \
--pretrained-teacher runs/celeba/baseline/0/res34/checkpoints/latest.pth \
--seed 0 \
--log-dir res34_res34 &

# CUDA_VISIBLE_DEVICES=7 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
# --student resnet34 \
# --teacher resnet34 \
# --loss adv --adv \
# --lambda_ 0.50 \
# --root runs/celeba/adv \
# --pretrained-teacher runs/celeba/baseline/1/res34/checkpoints/latest.pth \
# --seed 1 \
# --log-dir res34_res34 &

CUDA_VISIBLE_DEVICES=7 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--student resnet34 \
--teacher resnet34 \
--loss adv --adv \
--lambda_ 0.50 \
--root runs/celeba/adv \
--pretrained-teacher runs/celeba/baseline/2/res34/checkpoints/latest.pth \
--seed 2 \
--log-dir res34_res34 &

# seed=123
# CUDA_VISIBLE_DEVICES=2 python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
# --student resnet34 \
# --teacher resnet34 \
# --loss adv --adv \
# --lambda_ 0.50 \
# --pretrained-teacher runs/celeba/baseline/123/res34/checkpoints/latest.pth \
# --root runs/celeba/adv \
# --seed 123 \
# --log-dir res34_res34 &