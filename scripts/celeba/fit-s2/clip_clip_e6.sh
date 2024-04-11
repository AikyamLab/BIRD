# seed=42
python3 main.py --num-task-classes 2 --dataset celeba --epochs 4 --lr 0.0001 \
--student clip \
--teacher clip \
--fitnet-s2 --vlm clip \
--pretrained-teacher runs/celeba/baseline/42/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s2-e6 --epochs 4 --lr 0.0001 \
--continue-student runs/celeba/fit-s1-e6/42/clip_clip/checkpoints/latest.pth \
--seed 42 \
--log-dir clip_clip &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba --epochs 4 --lr 0.0001 \
--student clip \
--teacher clip \
--fitnet-s2 --vlm clip \
--root runs/celeba/fit-s2-e6 --epochs 4 --lr 0.0001 \
--continue-student runs/celeba/fit-s1-e6/0/clip_clip/checkpoints/latest.pth \
--pretrained-teacher runs/celeba/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_clip &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba --epochs 4 --lr 0.0001 \
--student clip \
--teacher clip \
--fitnet-s2 --vlm clip \
--pretrained-teacher runs/celeba/baseline/123/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s2-e6 --epochs 4 --lr 0.0001 \
--continue-student runs/celeba/fit-s1-e6/123/clip_clip/checkpoints/latest.pth \
--seed 123 \
--log-dir clip_clip &

# seed=1
python3 main.py --num-task-classes 2 --dataset celeba --epochs 4 --lr 0.0001 \
--student clip \
--teacher clip \
--fitnet-s2 --vlm clip \
--pretrained-teacher runs/celeba/baseline/1/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s2-e6 --epochs 4 --lr 0.0001 \
--continue-student runs/celeba/fit-s1-e6/1/clip_clip/checkpoints/latest.pth \
--seed 1 \
--log-dir clip_clip &

# seed=2
python3 main.py --num-task-classes 2 --dataset celeba --epochs 4 --lr 0.0001 \
--student clip \
--teacher clip \
--fitnet-s2 --vlm clip \
--pretrained-teacher runs/celeba/baseline/2/clip/checkpoints/latest.pth \
--root runs/celeba/fit-s2-e6 --epochs 4 --lr 0.0001 \
--continue-student runs/celeba/fit-s1-e6/2/clip_clip/checkpoints/latest.pth \
--seed 2 \
--log-dir clip_clip &