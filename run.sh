
python3 -m vit_jax.train --name ViT-B_16_`date +%F_%H%M%S` --model ViT-B_16 --logdir $logdir   --dataset imagenet2012   --batch 128  --warmup_steps 10000  --accum_steps 1 --shuffle_buffer 256000  --total_steps 93834 --base_lr 0.003 --decay_type linear  --eval_every 1000

