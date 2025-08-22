
import argparse, os, json, subprocess, sys, tempfile, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .prompt_utils import SYSTEM_COMPETITIVE, build_user_prompt

REPAIR_SYSTEM=("You are a senior Python developer. You will be given a program, an input, and the observed output or error. "
    "Return a corrected Python program that fixes the issue while still solving the original problem. "
    "Output only the final Python program, no explanations.")

def extract_code(text):
    m=re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S); return m[-1] if m else text


def run_python(code, stdin):
    with tempfile.TemporaryDirectory() as td:
        path=os.path.join(td,"main.py"); open(path,"w",encoding="utf-8").write(code)
        try:
            p=subprocess.run([sys.executable,path],input=stdin.encode("utf-8"),
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=3.0)
            return p.stdout.decode("utf-8",errors="replace"), p.stderr.decode("utf-8",errors="replace"), p.returncode
        except subprocess.TimeoutExpired: return "","TIMEOUT",-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subset",default="verifiable"); ap.add_argument("--split",default="test")
    ap.add_argument("--sample",type=int,default=50)
    ap.add_argument("--model_id",default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--max_new_tokens",type=int,default=1024); ap.add_argument("--out",default="outputs/repair_results.jsonl")
    args=ap.parse_args()
    ds=load_dataset("open-r1/codeforces",name=args.subset,split=args.split)
    if args.sample>0: 
        ds=ds.select(range(min(args.sample,len(ds))))

    tok=AutoTokenizer.from_pretrained(args.model_id,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",trust_remote_code=True)
    os.makedirs(os.path.dirname(args.out),exist_ok=True); fout=open(args.out,"w",encoding="utf-8")

    for row in ds:
        exs=row.get("examples") or []
        if not exs: 
            continue
        ex=exs[0]; gold_in=ex.get("input",""); gold_out=ex.get("output","").strip()
        messages=[{"role":"system","content":SYSTEM_COMPETITIVE},{"role":"user","content":build_user_prompt(row)}]
        prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen=model.generate(**tok(prompt,return_tensors="pt").to(model.device),
            max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=tok.eos_token_id)
        text=tok.decode(gen[0],skip_special_tokens=False); code=extract_code(text[len(prompt):])
        out,err,rc=run_python(code,gold_in); ok=(out.strip()==gold_out)
        repaired=False

        if not ok:
            repair_user=(f"Original Problem:\\n{build_user_prompt(row)}\\n\\nProgram:\\n```python\\n{code}\\n```\\n\\n"
                         f"Input:\\n{gold_in}\\n\\nObserved:\\n{out}\\nErrors:\\n{err}\\n\\nPlease return a corrected program.")
            rep_msgs=[{"role":"system","content":REPAIR_SYSTEM},{"role":"user","content":repair_user}]
            rp=tok.apply_chat_template(rep_msgs, tokenize=False, add_generation_prompt=True)
            gen2=model.generate(**tok(rp,return_tensors="pt").to(model.device),
                max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95,
                eos_token_id=tok.eos_token_id)
            text2=tok.decode(gen2[0],skip_special_tokens=False); code2=extract_code(text2[len(rp):])
            out2,err2,rc2=run_python(code2,gold_in); repaired=(out2.strip()==gold_out); ok=ok or repaired

        rec={"id":row.get("id"),"title":row.get("title",""),"pass_after_first":bool(ok and not repaired),"pass_after_repair":bool(repaired)}
        fout.write(json.dumps(rec)+"\n"); print(rec)
    
    fout.close(); print("Saved to", args.out)


if __name__=="__main__": main()
