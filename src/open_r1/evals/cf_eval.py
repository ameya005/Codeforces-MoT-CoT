
import argparse, csv, os, time, re, subprocess, sys, tempfile, json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .prompt_utils import SYSTEM_COMPETITIVE, build_user_prompt

def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    return m[-1] if m else text

def run_python(code: str, input_str: str, timeout=3.0):
    print(code)
    with tempfile.TemporaryDirectory() as td:
        prog = os.path.join(td, "main.py")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, prog], input=input_str.encode("utf-8"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        #print(p.stdout.decode("utf-8", errors="replace").strip())
       # print(p.stderr.decode("utf-8", errors="replace"))
            return p.stdout.decode("utf-8", errors="replace").strip(), p.stderr.decode("utf-8", errors="replace"), p.returncode
    
        except subprocess.TimeoutExpired:
             return "", "TIMEOUT", -1


def main():
    ap = argparse.ArgumentParser(description="Quick eval on Codeforces (HF) with CSV logging.")
    ap.add_argument("--subset", default="verifiable")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--model_id", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--csv", default="outputs/eval_log.csv")
    ap.add_argument("--save_generations", default="outputs/gens_eval_log.jsonl")
    ap.add_argument("--passes", type=int, default=1, help="Number of passes for pass@k evaluation (e.g., 1, 3, 5)")
    ap.add_argument("--checkpoint_path", default=None, help="Path to local checkpoint directory to load model from")
    ap.add_argument("--weights_path", default=None, help="Path to converted weights directory (use with base model_id)")
    args = ap.parse_args()

    processed_ids = set()
    if args.save_generations and os.path.exists(args.save_generations):
        with open(args.save_generations, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
        if processed_ids:
            print(f"Loaded {len(processed_ids)} previously processed samples from {args.save_generations}")

    ds = load_dataset("open-r1/codeforces", name=args.subset, split=args.split)
    
    initial_sample_size = len(ds)
    if processed_ids:
        ds = ds.filter(lambda example: example["id"] not in processed_ids)
        print(f"Filtered dataset to {len(ds)} samples (skipped {initial_sample_size - len(ds)} already processed samples).")

    if args.sample > 0: 
        ds = ds.select(range(min(args.sample, len(ds))))
    
    # Determine model source - local checkpoint, converted weights, or HuggingFace model
    if args.weights_path:
        # Load base model config from model_id, then load converted weights
        print(f"Loading base model config from: {args.model_id}")
        print(f"Loading converted weights from: {args.weights_path}")
        model_source = args.model_id
        weights_source = args.weights_path
    elif args.checkpoint_path:
        # Load complete model from local checkpoint
        model_source = args.checkpoint_path
        weights_source = None
    else:
        # Load from HuggingFace model
        model_source = args.model_id
        weights_source = None
    
    print(f"Loading model from: {model_source}")
    
    tok = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    
    if weights_source:
        # Load base model config, then load converted weights
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", 
            trust_remote_code=True,
        )
        # Load the converted weights (handle both single and sharded files)
        print(f"Loading converted weights from {weights_source}...")
        
        # Check if we have sharded weights or a single file
        single_weight_file = os.path.join(weights_source, "pytorch_model.bin")
        if os.path.exists(single_weight_file):
            # Single weight file
            state_dict = torch.load(single_weight_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("Single weight file loaded successfully!")
        else:
            # Sharded weight files - use the directory directly
            print("Detected sharded weight files, loading from directory...")
            model = AutoModelForCausalLM.from_pretrained(
                weights_source,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto", 
                trust_remote_code=True,
            )
            print("Sharded weights loaded successfully!")
    else:
        # Load complete model
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True,
        )
    
    os.makedirs(os.path.dirname(args.save_generations), exist_ok=True)
    gen_out_mode = "a" if os.path.exists(args.save_generations) else "w"
    gen_out = open(args.save_generations, gen_out_mode, encoding="utf-8")
    total=evaluable=solved=total_new_tokens=0
    t0=time.time()

    for row in ds:
        problem_solved = False
        user = build_user_prompt(row)
        messages=[{"role":"system","content":SYSTEM_COMPETITIVE},{"role":"user","content":user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        time_limit = row.get("time_limit", 3.0)
        num_test_cases = row.get("testset_size", 10)
        time_limit *= num_test_cases

        # Prepare batch of prompts for parallel generation
        batched_prompts = [prompt] * args.passes
        inputs = tok(batched_prompts, return_tensors="pt", padding=True).to(model.device)

        gen_ids_batch = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature>0.0, temperature=args.temperature, top_p=args.top_p,
            eos_token_id=tok.eos_token_id, repetition_penalty=1.05)

        # Process each generated pass
        for i in range(args.passes):
            gen_ids = gen_ids_batch[i]
            text = tok.decode(gen_ids, skip_special_tokens=False)
            gen_text = text[len(batched_prompts[i]):]
            code = extract_code(gen_text)
            new_tok = tok(gen_text, add_special_tokens=False, return_tensors=None)
            total_new_tokens += len(new_tok["input_ids"])
            rec={"id":row.get("id"),"title":row.get("title",""),"generation":gen_text}
            gen_out.write(json.dumps(rec, ensure_ascii=False)+"\n")
            official_tests=row.get("examples") or []
            if official_tests:
                ex=official_tests[0]
                out,err,rc = run_python(code, ex.get("input",""), time_limit)
                print(f"after run_python: {out}")
                print(f"expected output: {ex.get('output','').strip()}")
                if out.strip()==(ex.get("output","").strip()): 
                    problem_solved = True
        if problem_solved:
            solved+=1
        if official_tests: 
            evaluable+=1 
        
        total+=1
    gen_out.close()
    elapsed=time.time()-t0
    avg_new_tokens=(total_new_tokens/total) if total else 0.0
    pass_rate=(solved/evaluable) if evaluable else 0.0
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    header=["timestamp","model_id","subset","split","sample","temperature","top_p","max_new_tokens","passes",
            "problems_total","problems_evaluable","solved_on_example","example_pass_rate",
            "avg_new_tokens","elapsed_sec"]
    write_header=not os.path.exists(args.csv)
    with open(args.csv,"a",newline="",encoding="utf-8") as f:
        import csv; w=csv.writer(f)
        if write_header: 
            w.writerow(header)
        
        # Determine what to log as model_id
        if args.weights_path:
            logged_model_id = f"{args.model_id}+{args.weights_path}"
        elif args.checkpoint_path:
            logged_model_id = args.checkpoint_path
        else:
            logged_model_id = args.model_id
            
        w.writerow([datetime.utcnow().isoformat()+"Z", logged_model_id, args.subset, args.split, len(ds),
                    args.temperature, args.top_p, args.max_new_tokens, args.passes,
                    total, evaluable, solved, round(pass_rate,4), round(avg_new_tokens,2), round(elapsed,2)])
    print(f"[LOGGED] {solved}/{evaluable} example-pass -> {pass_rate:.3f}. Avg new toks: {avg_new_tokens:.1f}. Elapsed: {elapsed:.1f}s")
    print(f"CSV -> {args.csv}"); print(f"Generations -> {args.save_generations}")


if __name__ == "__main__": 
    main()
