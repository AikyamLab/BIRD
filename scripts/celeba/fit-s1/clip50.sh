# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip50 \
--teacher clip50 \
--fitnet-s1 --vlm clip50 \
--pretrained-teacher runs/celeba/baseline/42/clip50/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 42 \
--log-dir clip50_clip50 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip50 \
--teacher clip50 \
--fitnet-s1 --vlm clip50 \
--root runs/celeba/fit-s1 --epochs 2 \
--pretrained-teacher runs/celeba/baseline/0/clip50/checkpoints/latest.pth \
--seed 0 \
--log-dir clip50_clip50 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip50 \
--teacher clip50 \
--fitnet-s1 --vlm clip50 \
--pretrained-teacher runs/celeba/baseline/123/clip50/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 123 \
--log-dir clip50_clip50 &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip50 \
--teacher clip50 \
--fitnet-s1 --vlm clip50 \
--pretrained-teacher runs/celeba/baseline/1/clip50/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 1 \
--log-dir clip50_clip50 &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip50 \
--teacher clip50 \
--fitnet-s1 --vlm clip50 \
--pretrained-teacher runs/celeba/baseline/2/clip50/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 2 \
--log-dir clip50_clip50 &