i=0
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}


for seed in 0; do
  for rank in 128; do
    for optimizer_states in  "unchanged"; do #  "reset" "transform"
      for lr in 2e-4; do
        cuda_device=${gpus[$((i % num_gpus))]}
        echo "Running with seed=${seed} on CUDA device=${cuda_device}"
        echo "python centralized_glue_cp.py --lora_r ${rank}  --lora_alpha ${rank} --seed ${seed} --task cola --device ${cuda_device} --lr ${lr} --method rso --optim AdamW --interval 100 --optimizer_states ${optimizer_states} --wandb"
        # 实际运行命令
        python centralized_glue_cp.py --lora_r ${rank} --lora_alpha ${rank} --seed ${seed} --task cola --device ${cuda_device} --lr ${lr} --method rso --optim AdamW --interval 100 --optimizer_states ${optimizer_states} --wandb&
        ((i+=1))
      done
    done
  done
done

