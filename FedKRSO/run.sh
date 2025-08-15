i=0
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
rank=4 # set to rank=0 for full fine-tuning
kseed=10 # only work when agg
for alpha in -1; do # -1=iid, 0.5, 0.25
  for task in "qnli"; do #   "qnli" "cola" "sst2" "stsb" "mnli_mismatched"; do
    for method in "rso" "normal" "ffa"; do
      for interval in 10; do
        for batchsize in 32; do
          for seed in 0 1 2; do
            for lr in 2e-4; do
               cuda_device=${gpus[$((i % num_gpus))]}
              echo "Running with seed=${seed} on CUDA device=${cuda_device}"
              echo "python fed_train_glue_RSO.py --task ${task} --device ${cuda_device} --batch_size ${batchsize} --seed $((seed)) --lr ${lr} --agg_type ${method} --interval ${interval} --amp --lora_r ${rank} --alpha ${alpha} --kseed ${kssed} --wandb"
              # 实际运行命令
              python fed_train_glue_RSO.py --task ${task} --device ${cuda_device} --batch_size ${batchsize} --seed $((seed)) --lr ${lr} --agg_type ${method} --interval ${interval} --amp --lora_r ${rank} --alpha ${alpha} --kseed ${kseed} --wandb&
              ((i+=1))
            done
          done
        done
      done
    done
  done
done


#for seed in 0 1; do
#  for rank in 128; do
#    for optimizer_states in  "unchanged"; do #  "reset" "transform"
#      for lr in 2e-5 5e-5 1e-4 2e-4; do
#        cuda_device=$((i % gpu_count))
#        echo "Running with seed=${seed} on CUDA device=${cuda_device}"
#        echo "python train_LoRA_Primer.py --lora_r ${rank}  --lora_alpha ${rank} --seed ${seed} --task cola --device ${cuda_device} --lr ${lr} --method rso --optim AdamW --interval 100 --optimizer_states ${optimizer_states} --wandb"
#        # 实际运行命令
#        python train_LoRA_Primer.py --lora_r ${rank} --lora_alpha ${rank} --seed ${seed} --task cola --device ${cuda_device} --lr ${lr} --method rso --optim AdamW --interval 100 --optimizer_states ${optimizer_states} --wandb&
#        ((i+=1))
#      done
#    done
#  done
#done


#
#pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
#pip install scikit-learn==1.6.0
#pip install peft==0.12.0
#pip install pycocoevalcap==1.2
#pip install datasets==3.2.0
#pip install wandb==0.19.1
#pip install tqdm==4.66.4