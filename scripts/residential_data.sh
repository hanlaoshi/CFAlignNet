#!/bin/bash
# ============================================================
# CFAlignNet - Residential Load Dataset
# ============================================================

# Common parameters
model_name=CFAlignNet
train_epochs=100
learning_rate=0.0001
llama_layers=16

master_port=29589
num_process=2
batch_size=2
d_model=16
d_ff=32
llm_model=GPT2
llm_dim=768
percent=100
patch_len=16
patience=18
comment='CFAlignNet_residential_data'

export CUDA_VISIBLE_DEVICES=0,1

# Parameter arrays
pred_len_array=(720)
seq_len_array=(192)

# Create logs directory
mkdir -p logs/residential_data

# Loop through parameter combinations
for seq_len in "${seq_len_array[@]}"; do
  for pred_len in "${pred_len_array[@]}"; do
    label_len=$((seq_len / 2))

    log_file="logs/residential_data/CFAlignNet_seq${seq_len}_pred${pred_len}.txt"

    echo "Running with seq_len=$seq_len, pred_len=$pred_len, label_len=$label_len"

    accelerate launch --multi_gpu --num_processes $num_process --main_process_port $master_port \
      run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path residential_data_selected.csv \
      --model_id CFAlignNet \
      --model $model_name \
      --data Residential_data \
      --llm_dim $llm_dim \
      --patience $patience \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --percent $percent \
      --llm_model $llm_model \
      --patch_len $patch_len \
      --factor 3 \
      --enc_in 3 \
      --dec_in 3 \
      --c_out 3 \
      --des 'Exp' \
      --itr 1 \
      --holiday_data_path ./dataset/holiday_data_transform.xlsx \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --llm_layers $llama_layers \
      --train_epochs $train_epochs \
      --model_comment "${comment}_seq${seq_len}_pred${pred_len}" | tee $log_file

    echo "Completed run with seq_len=$seq_len, pred_len=$pred_len"
    echo "Log saved to $log_file"
    echo "------------------------"

    sleep 2
  done
done

echo "All experiments completed."
