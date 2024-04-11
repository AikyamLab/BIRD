# seed=42
python3 main.py --num-task-classes 2 --dataset celeba \
--student clip \
--teacher clip --vlm clip \
--loss adv --adv \
--lambda_ 1 \
--pretrained-teacher runs/celeba/baseline/42/clip/checkpoints/latest.pth \
--root runs/celeba/adv --epochs 10 \
--seed 42 \
--log-dir clip_clip &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba \
--student clip \
--teacher clip --vlm clip \
--loss adv --adv \
--lambda_ 1 \
--root runs/celeba/adv --epochs 10 \
--pretrained-teacher runs/celeba/baseline/0/clip/checkpoints/latest.pth \
--seed 0 \
--log-dir clip_clip &

python3 main.py --num-task-classes 2 --dataset celeba \
--student clip \
--teacher clip --vlm clip \
--loss adv --adv \
--lambda_ 1 \
--root runs/celeba/adv --epochs 10 \
--pretrained-teacher runs/celeba/baseline/1/clip/checkpoints/latest.pth \
--seed 1 \
--log-dir clip_clip &

python3 main.py --num-task-classes 2 --dataset celeba \
--student clip \
--teacher clip --vlm clip \
--loss adv --adv \
--lambda_ 1 \
--root runs/celeba/adv --epochs 10 \
--pretrained-teacher runs/celeba/baseline/2/clip/checkpoints/latest.pth \
--seed 2 \
--log-dir clip_clip &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba \
--student clip \
--teacher clip --vlm clip \
--loss adv --adv \
--lambda_ 1 \
--pretrained-teacher runs/celeba/baseline/123/clip/checkpoints/latest.pth \
--root runs/celeba/adv --epochs 10 \
--seed 123 \
--log-dir clip_clip &