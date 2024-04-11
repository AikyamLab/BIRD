# seed=42
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip50 \
--vlm clip50 \
--root runs/celeba/baseline \
--seed 42 \
--lr 1e-3 --epochs 10 \
--log-dir clip50 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip50 \
--vlm clip50 \
--root runs/celeba/baseline \
--seed 0 \
--lr 1e-3 --epochs 10 \
--log-dir clip50 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip50 \
--vlm clip50 \
--root runs/celeba/baseline \
--seed 123 \
--lr 1e-3 --epochs 10 \
--log-dir clip50 &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip50 \
--vlm clip50 \
--root runs/celeba/baseline \
--seed 1 \
--lr 1e-3 --epochs 10 \
--log-dir clip50 &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba \
--base clip50 \
--vlm clip50 \
--root runs/celeba/baseline \
--seed 2 \
--lr 1e-3 --epochs 10 \
--log-dir clip50 &