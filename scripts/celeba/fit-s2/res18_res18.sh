# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student resnet18 \
--teacher resnet18 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/42/res18/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/42/res18_res18/checkpoints/latest.pth \
--seed 42 \
--log-dir res18_res18 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student resnet18 \
--teacher resnet18 \
--fitnet-s2 \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/res18_res18/checkpoints/latest.pth \
--pretrained-teacher runs/celeba/baseline/0/res18/checkpoints/latest.pth \
--seed 0 \
--log-dir res18_res18 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student resnet18 \
--teacher resnet18 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/123/res18/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/123/res18_res18/checkpoints/latest.pth \
--seed 123 \
--log-dir res18_res18 &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student resnet18 \
--teacher resnet18 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/1/res18/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/1/res18_res18/checkpoints/latest.pth \
--seed 1 \
--log-dir res18_res18 &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student resnet18 \
--teacher resnet18 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/2/res18/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/2/res18_res18/checkpoints/latest.pth \
--seed 2 \
--log-dir res18_res18 &