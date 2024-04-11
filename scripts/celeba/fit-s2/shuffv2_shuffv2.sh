# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/42/shuffv2/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/42/shuffv2_shuffv2/checkpoints/latest.pth \
--seed 42 \
--log-dir shuffv2_shuffv2 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--fitnet-s2 \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/0/shuffv2_shuffv2/checkpoints/latest.pth \
--pretrained-teacher runs/celeba/baseline/0/shuffv2/checkpoints/latest.pth \
--seed 0 \
--log-dir shuffv2_shuffv2 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/123/shuffv2/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/123/shuffv2_shuffv2/checkpoints/latest.pth \
--seed 123 \
--log-dir shuffv2_shuffv2 &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/1/shuffv2/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/1/shuffv2_shuffv2/checkpoints/latest.pth \
--seed 1 \
--log-dir shuffv2_shuffv2 &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 8 --lr 0.0001 \
--student shufflenetv2 \
--teacher shufflenetv2 \
--fitnet-s2 \
--pretrained-teacher runs/celeba/baseline/2/shuffv2/checkpoints/latest.pth \
--root runs/celeba/fit-s2 --epochs 8 --lr 0.0001 \
--continue-student runs/celeba/fit-s1/2/shuffv2_shuffv2/checkpoints/latest.pth \
--seed 2 \
--log-dir shuffv2_shuffv2 &