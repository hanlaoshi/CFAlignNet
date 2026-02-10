#!/bin/bash
# ============================================================
# CFAlignNet - ETTh2 Dataset
# ============================================================

# Common parameters
model_name=CFAlignNet
train_epochs=100

master_port=29519
num_process=2
d_model=16
d_ff=32
llm_model=GPT2
llm_dim=768
percent=100
patience=5
comment='CFAlignNet_etth2'

export CUDA_VISIBLE_DEVICES=0,1

# Parameter arrays
pred_len_array=(192)
seq_len_array=(192)
learning_rate_array=(0.0001)
batch_size_array=(8)
patch_len_array=(24)
llama_layers_array=(6)
lora_r_array=(8)
lora_alpha_array=(8)
lora_dropout_array=(0.1)

# Create logs directory
mkdir -p logs/etth2

# Initialize experiment counter
experiment_count=0
total_experiments=$((${#seq_len_array[@]} * ${#pred_len_array[@]} * ${#learning_rate_array[@]} * ${#batch_size_array[@]} * ${#patch_len_array[@]} * ${#llama_layers_array[@]} * ${#lora_r_array[@]} * ${#lora_alpha_array[@]} * ${#lora_dropout_array[@]}))

echo "Starting parameter sweep with $total_experiments total experiments..."
echo "========================"

# Loop through parameter combinations
for seq_len in "${seq_len_array[@]}"; do
  for pred_len in "${pred_len_array[@]}"; do
    for learning_rate in "${learning_rate_array[@]}"; do
      for batch_size in "${batch_size_array[@]}"; do
        for patch_len in "${patch_len_array[@]}"; do
          for llama_layers in "${llama_layers_array[@]}"; do
            for lora_r in "${lora_r_array[@]}"; do
              for lora_alpha in "${lora_alpha_array[@]}"; do
                for lora_dropout in "${lora_dropout_array[@]}"; do
                  experiment_count=$((experiment_count + 1))
                  label_len=$((seq_len / 2))

                  lr_formatted=$(echo $learning_rate | sed 's/\./_/g')
                  lora_dropout_formatted=$(echo $lora_dropout | sed 's/\./_/g')
                  log_file="logs/etth2/exp${experiment_count}_seq${seq_len}_pred${pred_len}_lr${lr_formatted}_bs${batch_size}_patch${patch_len}_layers${llama_layers}_lorar${lora_r}_loraalpha${lora_alpha}.txt"

                  echo "[$experiment_count/$total_experiments] Running experiment..."

                  accelerate launch --multi_gpu --num_processes $num_process --main_process_port $master_port \
                    run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ./dataset/ \
                    --data_path ETTh2.csv \
                    --model_id CFAlignNet \
                    --model $model_name \
                    --data ETTh2 \
                    --llm_dim $llm_dim \
                    --patience $patience \
                    --features M \
                    --freq h \
                    --seq_len $seq_len \
                    --label_len $label_len \
                    --pred_len $pred_len \
                    --percent $percent \
                    --llm_model $llm_model \
                    --patch_len $patch_len \
                    --factor 7 \
                    --enc_in 7 \
                    --dec_in 7 \
                    --c_out 7 \
                    --des 'Exp' \
                    --itr 1 \
                    --holiday_data_path ./dataset/holiday_data_etth.xlsx \
                    --d_model $d_model \
                    --d_ff $d_ff \
                    --batch_size $batch_size \
                    --learning_rate $learning_rate \
                    --llm_layers $llama_layers \
                    --train_epochs $train_epochs \
                    --lora_r $lora_r \
                    --lora_alpha $lora_alpha \
                    --lora_dropout $lora_dropout \
                    --model_comment "${comment}_seq${seq_len}_pred${pred_len}_lr${lr_formatted}_bs${batch_size}" | tee $log_file

                  if [ $? -eq 0 ]; then
                    echo "Experiment $experiment_count completed successfully"
                  else
                    echo "Experiment $experiment_count failed"
                  fi

                  echo "Progress: $experiment_count/$total_experiments"
                  echo "========================"

                  sleep 2
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All $total_experiments experiments completed."
