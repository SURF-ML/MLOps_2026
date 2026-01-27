import asyncio
import json
import argparse
import os
import pickle
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

async def process_item(client, item, semaphore, args):
    """Processes a single text for embedding and returns (pid, embedding)."""
    pid = item.get("pid", "unknown_pid")
    
    # Priority for text extraction: 'report' (new), 'response', etc.
    raw_text = item.get("report") or item.get("response") or item.get("text") or item.get("prompt") or ""
    
    if args.is_query and raw_text:
        final_input = get_detailed_instruct(args.task_description, raw_text)
    else:
        final_input = raw_text

    async with semaphore:
        try:
            response = await client.embeddings.create(
                model=args.model,
                input=final_input,
                timeout=60
            )
            return pid, response.data[0].embedding
        except Exception as e:
            print(f"\nError on PID {pid}: {e}")
            return pid, None

async def main(args):
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-embed")
    semaphore = asyncio.Semaphore(args.concurrency)

    # 1. Load existing dictionary for Resume
    results_dict = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, 'rb') as f:
                results_dict = pickle.load(f)
            print(f"Resuming: Found {len(results_dict)} existing PIDs.")
        except (EOFError, pickle.UnpicklingError):
            print("Warning: Existing pkl file was corrupted or empty. Starting fresh.")

    # 2. Load input JSONL and filter out already processed PIDs
    remaining_data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pid = item.get("pid")
                if pid and pid not in results_dict:
                    remaining_data.append(item)

    if not remaining_data:
        print("All PIDs already processed. Success!")
        return

    # 3. Run Tasks
    print(f"Embedding {len(remaining_data)} items individually...")
    tasks = [process_item(client, item, semaphore, args) for item in remaining_data]
    
    # Use as_completed to save as we go (stream-to-dict)
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Encoding"):
        pid, embedding = await future
        if embedding is not None:
            # We convert to float32 for smaller pkl size
            results_dict[pid] = np.array(embedding, dtype=np.float32)
        else:
            print(f"Skipping saving for failed PID: {pid}")

    # 4. Final Save (Pickle is more flexible than npz for dictionaries)
    with open(args.output, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Done! Total unique PIDs in {args.output}: {len(results_dict)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input .jsonl with 'pid' and 'report'")
    parser.add_argument("--output", type=str, required=True, help="Output .pkl file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--is_query", action="store_true")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--task_description", type=str, default="Retrieve relevant medical passages")
    args = parser.parse_args()
    asyncio.run(main(args))
