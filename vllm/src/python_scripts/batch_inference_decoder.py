import asyncio
import json
import argparse
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

async def process_item(client, item, args, semaphore):
    # Determine mode-specific parameters
    # Note: These smaller models often use system prompt prefixes for mode switching
    raw_text = item.get("report") or item.get("response") or item.get("text") or item.get("prompt") or ""
    if args.mode == "think":
        final_prompt = f"/think {raw_text}"
        rec_temp, rec_top_p, rec_top_k = 0.6, 0.95, 20
    else:
        final_prompt = f"/no_think {raw_text}"
        rec_temp, rec_top_p, rec_top_k = 0.7, 0.8, 20

    # Apply overrides if provided
    temp = args.temperature if args.temperature is not None else rec_temp
    top_p = args.top_p if args.top_p is not None else rec_top_p
    top_k = args.top_k if args.top_k is not None else rec_top_k

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="Qwen/Qwen3-4B-AWQ",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=temp,
                top_p=top_p,
                extra_body={"top_k": top_k, "min_p": 0.0},
                max_tokens=args.max_tokens,
                timeout=180 # 3-minute timeout if something doesn't work, 
            )
            # Keeping original fields + adding response
            item["report"] = response.choices[0].message.content
            item["mode"] = args.mode
            item["finish_reason"] = response.choices[0].finish_reason
        except Exception as e:
            item["error"] = str(e)
            item["report"] = None
        return item

async def run_batch(args):
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm")
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # 1. Load already processed Prompts to support Resume
    processed_prompts = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    if 'prompt' in res:
                        processed_prompts.add(res['prompt'])
                except:
                    continue
        print(f"Resuming: Found {len(processed_prompts)} items already processed.")

    # 2. Load input data safely, ignoring empty lines and duplicates
    data = []
    print(f"Reading input from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                item = json.loads(clean_line)
                if item.get('prompt') not in processed_prompts:
                    data.append(item)

    if not data:
        print("All items already processed or input is empty.")
        return

    # 3. Process and write to file immediately (Stream-to-disk)
    print(f"Processing {len(data)} items...")
    tasks = [process_item(client, item, args, semaphore) for item in data]
    
    # Use 'a' (append) mode
    with open(args.output, 'a', encoding='utf-8') as f:
        # as_completed loop for immediate writing
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inferencing 4B"):
            result = await future
            # ensure_ascii=False keeps Unicode characters (like non-English text) readable
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush() # Force write to disk

    print(f"Batch complete. Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Batch Inference for Qwen-4B")
    parser.add_argument("--input", type=str, required=True, help="Input .jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", type=str, choices=["think", "no_think"], default="no_think")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    
    args = parser.parse_args()
    asyncio.run(run_batch(args))
