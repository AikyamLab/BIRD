# seed=42
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip \
--vlm clip \
--root runs/celeba/baseline \
--seed 42 \
--lr 1e-3 --epochs 10 \
--log-dir clip &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip \
--vlm clip \
--root runs/celeba/baseline \
--seed 0 \
--lr 1e-3 --epochs 10 \
--log-dir clip &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip \
--vlm clip \
--root runs/celeba/baseline \
--seed 123 \
--lr 1e-3 --epochs 10 \
--log-dir clip &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip \
--vlm clip \
--root runs/celeba/baseline \
--seed 1 \
--lr 1e-3 --epochs 10 \
--log-dir clip &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip \
--vlm clip \
--root runs/celeba/baseline \
--seed 2 \
--lr 1e-3 --epochs 10 \
--log-dir clip &