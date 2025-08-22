eval-log:
	python -m src.open_r1.evals.cf_eval --sample 200 --passes 1 --save_generations outputs/gens_eval_log_pass1_promptv0.3.jsonl
	python -m src.open_r1.evals.cf_eval --sample 200 --passes 1 --save_generations outputs/gens_eval_log_pass1_promptv0.3.jsonl --passes 3
	python -m src.open_r1.evals.cf_eval --sample 200 --passes 1 --save_generations outputs/gens_eval_log_pass1_promptv0.3.jsonl --passes 5

eval-checkpoint:
	# Use the latest checkpoint directory (contains the actual model weights)
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.open_r1.evals.cf_eval \
	--sample 200 --passes 1 \
	--checkpoint_path  /root/ameya/storage/ameya_ckpts/checkpoints/qwen2_7b_merged_lora_kl_v2/checkpoint-336 \
	--save_generations outputs/gens_eval_checkpoint_sft_cot_short-330_prompt2-lora-assistant_only_ds_kl.jsonl
	

convert-checkpoint:
	# Convert to storage mount with more space
	python scripts/utils.py \
	--checkpoint_path ~/workspace/ameya_ckpts/checkpoints/qwen2-7b-full/checkpoint-100 \
	--output_dir storage/qwen2-7b-full-converted

convert-checkpoint-custom:
	@echo "Usage: make convert-checkpoint-custom CHECKPOINT=path/to/checkpoint OUTPUT=path/to/output"
	@echo "Example: make convert-checkpoint-custom CHECKPOINT=checkpoints/qwen2-7b-full/checkpoint-50 OUTPUT=checkpoints/qwen2-7b-full-converted-50"
	python scripts/utils.py \
	--checkpoint_path $(CHECKPOINT) \
	--output_dir $(OUTPUT)

eval-converted:
	python -m src.open_r1.evals.cf_eval \
  --sample 10 --passes 1 \
  --checkpoint_path /root/ameya/storage/ameya_ckpts/checkpoints/qwen2-7b-full-sft/checkpoint-10 \
  --save_generations outputs/gens_eval_converted_ckpt_50.jsonl


sft-lora:
	accelerate launch --num_processes 8 scripts/sft_train.py \
	--dataset_name /root/ameya/data/mot_r1_messages/all_data.jsonl \
	--custom_dataset \
	--model_id Qwen/Qwen2-7B-Instruct \
	--use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
	--max_seq_len 8192 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 8 \
	--learning_rate 1e-4 --num_train_epochs 3 \
	--packing --gradient_checkpointing \
	--report_to wandb --run_name qwen2-7b-lora-4k-ds \
	--deepspeed scripts/configs/ds_z2.json \
	--output_dir ~/ameya/storage/ameya_ckpts/checkpoints/qwen2-7b-lora-4k-ds_v2


sft-full-ds:
	@echo "Starting full-parameter SFT with Accelerate+DeepSpeed…"
	WANDB_MODE=online HF_HUB_ENABLE_HF_TRANSFER=1 \
	accelerate launch --num_processes 8 scripts/sft_train.py \
	  --dataset_name /root/ameya/data/mot_r1_messages/all_data.jsonl \
	  --custom_dataset \
	  --dataset_subset code \
	  --split train \
	  --model_id Qwen/Qwen2-7B-Instruct \
	  --max_seq_len 8192 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
	  --learning_rate 1e-5 --num_train_epochs 3 --gradient_checkpointing \
	  --report_to wandb --run_name qwen2-7b-full-8k-ds \
	  --deepspeed scripts/configs/ds_z3.json \
	  --output_dir ~/ameya/storage/ameya_ckpts/checkpoints/qwen2-7b-full-sft-preprocessed_v3 



kl-sft-lora:
	CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 scripts/sft_train_kl.py \
	--dataset_name /root/ameya/data/mot_r1_messages/all_data.jsonl \
	--custom_dataset \
	--model_id Qwen/Qwen2-7B-Instruct \
	--use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
	--max_seq_len 8192 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 8 \
	--learning_rate 1e-4 --num_train_epochs 3 \
	--packing --gradient_checkpointing \
	--report_to wandb --run_name qwen2-7b-lora-4k-ds \
	--deepspeed scripts/configs/ds_z2.json \
	--output_dir ~/ameya/storage/ameya_ckpts/checkpoints/qwen2-7b-lora-4k-kl \
	--kl_coef 0.00 \
	--kl_start 0 \
	--kl_t 1.0

