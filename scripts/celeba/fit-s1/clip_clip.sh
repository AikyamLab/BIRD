# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip \
--teacher clip \
--fitnet-s1 --vlm clip \
--pretrained-teacher runs/celeba/baseline/42/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 42 \
--log-dir clip_clip &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip \
--teacher clip \
--fitnet-s1 --vlm clip \
--root runs/celeba/fit-s1 --epochs 2 \
--pretrained-teacher runs/celeba/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_clip &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip \
--teacher clip \
--fitnet-s1 --vlm clip \
--pretrained-teacher runs/celeba/baseline/123/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 123 \
--log-dir clip_clip &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip \
--teacher clip \
--fitnet-s1 --vlm clip \
--pretrained-teacher runs/celeba/baseline/1/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 1 \
--log-dir clip_clip &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 2 \
--student clip \
--teacher clip \
--fitnet-s1 --vlm clip \
--pretrained-teacher runs/celeba/baseline/2/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s1 --epochs 2 \
--seed 2 \
--log-dir clip_clip &