# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--base resnet18 \
--root runs/celeba/baseline \
--seed 42 \
--log-dir res18 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--base resnet18 \
--root runs/celeba/baseline \
--seed 0 \
--log-dir res18 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--base resnet18 \
--root runs/celeba/baseline \
--seed 123 \
--log-dir res18 &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--base resnet18 \
--root runs/celeba/baseline \
--seed 1 \
--log-dir res18 &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 10 \
--base resnet18 \
--root runs/celeba/baseline \
--seed 2 \
--log-dir res18 &