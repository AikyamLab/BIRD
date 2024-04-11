# seed=42
python3 main.py --num-task-classes 2 --dataset celeba \
--student clip50 \
--teacher clip50 --vlm clip50 \
--loss adv --adv \
--lambda_ 1 \
--pretrained-teacher runs/celeba/baseline/42/clip50/checkpoints/latest.pth \
--root runs/celeba/adv --epochs 10 \
--seed 42 \
--log-dir clip50_clip50 &

# seed=0
python3 main.py --num-task-classes 2 --dataset celeba \
--student clip50 \
--teacher clip50 --vlm clip50 \
--loss adv --adv \
--lambda_ 1 \
--root runs/celeba/adv --epochs 10 \
--pretrained-teacher runs/celeba/baseline/0/clip50/checkpoints/latest.pth \
--seed 0 \
--log-dir clip50_clip50 &

python3 main.py --num-task-classes 2 --dataset celeba \
--student clip50 \
--teacher clip50 --vlm clip50 \
--loss adv --adv \
--lambda_ 1 \
--root runs/celeba/adv --epochs 10 \
--pretrained-teacher runs/celeba/baseline/1/clip50/checkpoints/latest.pth \
--seed 1 \
--log-dir clip50_clip50 &

python3 main.py --num-task-classes 2 --dataset celeba \
--student clip50 \
--teacher clip50 --vlm clip50 \
--loss adv --adv \
--lambda_ 1 \
--root runs/celeba/adv --epochs 10 \
--pretrained-teacher runs/celeba/baseline/2/clip50/checkpoints/latest.pth \
--seed 2 \
--log-dir clip50_clip50 &

# seed=123
python3 main.py --num-task-classes 2 --dataset celeba \
--student clip50 \
--teacher clip50 --vlm clip50 \
--loss adv --adv \
--lambda_ 1 \
--pretrained-teacher runs/celeba/baseline/123/clip50/checkpoints/latest.pth \
--root runs/celeba/adv --epochs 10 \
--seed 123 \
--log-dir clip50_clip50 &