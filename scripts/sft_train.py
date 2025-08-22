#!/usr/bin/env python
import argparse, os, time
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
import wandb
import glob

SYSTEM = ("You are a competitive programming assistant.")

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number and return the latest
    checkpoint_numbers = [int(os.path.basename(d).split("-")[-1]) for d in checkpoint_dirs]
    latest_checkpoint = max(checkpoint_numbers)
    return os.path.join(output_dir, f"checkpoint-{latest_checkpoint}")

def ensure_system(msgs):
    if len(msgs) and msgs[0].get("role") == "system":
        return msgs
    return [{"role":"system","content":SYSTEM}] + msgs

# def messages_to_text(tokenizer, msgs):
#     return tokenizer.apply_chat_template(
#         msgs, tokenize=False, add_generation_prompt=False
#     )


def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_world_size()
        except Exception:
            pass
    return int(os.environ.get("WORLD_SIZE", "1"))


class TokensPerSecCallback(TrainerCallback):
    """
    Estimates tokens/sec using:
        tokens_per_optimizer_step ≈ max_seq_len * per_device_train_batch_size * world_size * grad_accum_steps
    Logs:
        - tokens_per_sec (instantaneous since last log)
        - tokens_per_sec_avg (since training began)
    Note: With packing or variable sequence lengths, this is an approximation (industry standard).
    """
    def __init__(self, tokens_per_step: float, use_wandb: bool):
        self.tokens_per_step = float(tokens_per_step)
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # noqa
                self._wandb = wandb
            except Exception:
                self._wandb = None
        self.reset()

    def reset(self):
        self.start_time = None
        self.last_time = None
        self.last_step = 0
        self.total_tokens = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.start_time = now
        self.last_time = now
        self.last_step = 0
        self.total_tokens = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Trainer provides logs dict for this event
        if state.global_step is None:
            return
        now = time.perf_counter()
        step = int(state.global_step)
        step_delta = step - self.last_step
        if step_delta <= 0:
            self.last_time = now
            return

        elapsed = max(now - self.last_time, 1e-9)
        inst_tokens = self.tokens_per_step * step_delta
        inst_tps = inst_tokens / elapsed

        self.total_tokens += inst_tokens
        avg_elapsed = max(now - self.start_time, 1e-9)
        avg_tps = self.total_tokens / avg_elapsed

        # Attach to logs so Trainer/W&B/TensorBoard pick it up
        logs = logs or {}
        logs["tokens_per_sec"] = inst_tps
        logs["tokens_per_sec_avg"] = avg_tps

        # Also push directly to W&B to be safe (won't duplicate keys visually)
        if self._wandb is not None:
            try:
                self._wandb.log({"tokens_per_sec": inst_tps,
                                 "tokens_per_sec_avg": avg_tps}, step=step)
            except Exception:
                pass

        # Update internal counters
        self.last_time = now
        self.last_step = step

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--dataset_name", default="open-r1/Mixture-of-Thoughts")
    ap.add_argument("--dataset_subset", default="code")
    ap.add_argument("--split", default="train")
    # Model
    ap.add_argument("--model_id", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    # Optim
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    # Logging / saving
    ap.add_argument("--output_dir", default="checkpoints/qwen2-7b-sft")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--report_to", default="wandb", help="wandb|tensorboard|none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--packing", action="store_true", help="pack multiple samples per sequence")
    ap.add_argument("--run_name", default=None, help="W&B run name / HF Trainer run_name")
    # W&B specifics (optional)
    ap.add_argument("--wandb_project", default="qwen-cot-finetune")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_tags", default=None, help="comma or space separated")
    # PEFT / LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # QLoRA (4-bit)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--bnb_4bit_quant_type", default="nf4", choices=["nf4","fp4"])
    ap.add_argument("--bnb_4bit_compute_dtype", default="bfloat16", choices=["bfloat16","float16"])
    # Distributed
    ap.add_argument("--deepspeed", default=None, help="path to deepspeed json (optional)")
    ap.add_argument("--resume_ckpt", default=False, help="resume from checkpoint")
    ap.add_argument("--custom_dataset", action="store_true", help="Use if dataa is from a local preprocessed source")

    args = ap.parse_args()

    run = wandb.init(
        entity="ameyaj005",
        project="qwen-cot-finetune",
    )

    # Dataset
    if args.custom_dataset:
        ds = load_dataset("json", data_files={"train":args.dataset_name}, cache_dir="data/hf_cache_mot_r1_v1")["train"]
    else:
        ds = load_dataset(args.dataset_name, name=args.dataset_subset, split=args.split)
    # Leave a small margin for special tokens added by the chat template
    MAX_CTX = args.max_seq_len
    ds_filt = ds.filter(lambda r: r.get("num_tokens_serialized", 0) <= MAX_CTX+128)
    print(len(ds), "→", len(ds_filt))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # def _map(ex):
    #     msgs = ensure_system(ex["messages"])
    #     return {"text": messages_to_text(tokenizer, msgs)}
    
    # ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "text"])

    # Model (+ optional 4-bit)
    quant_cfg = None
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Check if we're running in distributed mode
    is_distributed = get_world_size() > 1
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto" if not is_distributed else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_cfg,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA
    peft_cfg = None
    if args.use_lora:
        targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none"
        )

    # Trainer config
    run_name = args.run_name or f"sft-{os.path.basename(args.output_dir)}"
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,             # keep last few, avoid disk bloat
        save_safetensors=True, 
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        packing=args.packing,
        deepspeed=args.deepspeed,
        run_name=run_name,
        assistant_only_loss=True, # Enabled because dataset is a conversational dataset.
        remove_unused_columns=False,         # <-- keep "messages"
        chat_template_path="HuggingFaceTB/SmolLM3-3B",
        eos_token="<|im_end|>" # needed if you use the assistant_only_loss
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_filt,
        args=sft_cfg,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )

    # tokens/sec callback
    world_size = get_world_size()
    tokens_per_step = (
        args.max_seq_len
        * args.per_device_train_batch_size
        * max(world_size, 1)
        * max(args.gradient_accumulation_steps, 1)
    )
    trainer.add_callback(
        TokensPerSecCallback(tokens_per_step=tokens_per_step,
                             use_wandb=("wandb" in (args.report_to or "").lower()))
    )

    trainer.train(resume_from_checkpoint=args.resume_ckpt if args.resume_ckpt else None)
    
    # Save the final model in HuggingFace format
    if args.use_lora:
        # For LoRA, save the merged model
        trainer.save_model(args.output_dir)
    else:
        # For full fine-tuning, save the model directly
        trainer.model.save_pretrained(args.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
