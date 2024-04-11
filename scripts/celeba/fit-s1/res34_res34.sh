# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student resnet34 \
--teacher resnet34 \
--fitnet-s1 \
--pretrained-teacher runs/celeba/baseline/42/res34/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 42 \
--log-dir res34_res34 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student resnet34 \
--teacher resnet34 \
--fitnet-s1 \
--root runs/celeba/fit-s1 --epochs 2 \
--pretrained-teacher runs/celeba/baseline/0/res34/checkpoints/latest.pth \
--seed 0 \
--log-dir res34_res34 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student resnet34 \
--teacher resnet34 \
--fitnet-s1 \
--pretrained-teacher runs/celeba/baseline/123/res34/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 123 \
--log-dir res34_res34 &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student resnet34 \
--teacher resnet34 \
--fitnet-s1 \
--pretrained-teacher runs/celeba/baseline/1/res34/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 1 \
--log-dir res34_res34 &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student resnet34 \
--teacher resnet34 \
--fitnet-s1 \
--pretrained-teacher runs/celeba/baseline/2/res34/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 2 \
--log-dir res34_res34 &