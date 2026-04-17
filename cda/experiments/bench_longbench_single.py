"""Single-task LongBench: run one task with N samples."""
import os, sys, gc, json, argparse, torch
sys.path.insert(0, os.path.expanduser("~/ing/research/cda"))
sys.path.insert(0, os.path.expanduser("~/ing/research"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from core.core.compression import HadamardQuantCompressor
from core.core.compressed_generate import _compress_kv_cache, manual_decode_step

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--output", default="runs/longbench_50")
args = parser.parse_args()

device = "cuda:0"; D = 128
DATA_DIR = os.path.expanduser("~/ing/research/data/longbench/data")
MAX_LEN = 7500; MAX_NEW = 50

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
    device_map=device, low_cpu_mem_usage=True).eval()

c4 = HadamardQuantCompressor(dim=D, bit_width=4, half_rotation=True)
c2 = HadamardQuantCompressor(dim=D, bit_width=2, half_rotation=True)

def f1_score(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens: return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0: return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gt_tokens)
    return 2 * prec * rec / (prec + rec)

def generate(model, tokenizer, prompt, k_comp, v_comp, max_new=MAX_NEW):
    enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
    ctx = enc.input_ids[:, :-1]; nxt = enc.input_ids[:, -1:]
    ctx_len = ctx.shape[1]
    with torch.no_grad():
        out = model(ctx, use_cache=True, return_dict=True)
        kv = out.past_key_values
        compressed = _compress_kv_cache(kv, k_comp, v_comp, skip_sinks=4)
        del kv, out; gc.collect(); torch.cuda.empty_cache()
        tokens = []
        for step in range(max_new):
            logits = manual_decode_step(model, nxt, compressed, k_comp, v_comp, ctx_len + step)
            nxt = torch.argmax(logits[:, -1:], dim=-1)
            tokens.append(nxt.item())
            if nxt.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(tokens, skip_special_tokens=True)

def generate_fp16(model, tokenizer, prompt, max_new=MAX_NEW):
    enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(enc.input_ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

task = args.task
fpath = os.path.join(DATA_DIR, f"{task}.jsonl")
with open(fpath) as f:
    data = [json.loads(l) for l in f][:args.samples]

print(f"Task: {task}, samples: {len(data)}")
fp16_f1s, cda_f1s = [], []
for i, item in enumerate(data):
    prompt = item.get("input", item.get("context", ""))
    answers = item.get("answers", [item.get("answer", "")])
    if isinstance(answers, str): answers = [answers]
    
    fp16_out = generate_fp16(model, tokenizer, prompt)
    cda_out = generate(model, tokenizer, prompt, c4, c2)
    
    fp16_f1 = max(f1_score(fp16_out, a) for a in answers) * 100
    cda_f1 = max(f1_score(cda_out, a) for a in answers) * 100
    fp16_f1s.append(fp16_f1); cda_f1s.append(cda_f1)
    
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(data)}: FP16={sum(fp16_f1s)/len(fp16_f1s):.1f}, CDA={sum(cda_f1s)/len(cda_f1s):.1f}")
    gc.collect(); torch.cuda.empty_cache()

result = {
    "task": task, "n_samples": len(data),
    "fp16_f1": round(sum(fp16_f1s)/len(fp16_f1s), 2),
    "cda_f1": round(sum(cda_f1s)/len(cda_f1s), 2),
    "retention": round(sum(cda_f1s)/max(sum(fp16_f1s),1e-9)*100, 1),
}
print(f"\n{task}: FP16={result['fp16_f1']:.1f}, CDA={result['cda_f1']:.1f}, retention={result['retention']:.0f}%")

os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, f"{task}.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {args.output}/{task}.json")
