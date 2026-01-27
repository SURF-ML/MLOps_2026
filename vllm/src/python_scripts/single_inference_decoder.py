import argparse
import sys
from openai import OpenAI

def run_single_inference(args):
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="token-vllm")
    
    # 1. Determine base recommended parameters based on mode
    if args.mode == "think":
        final_prompt = f"/think {args.prompt}"
        rec_temp, rec_top_p, rec_top_k = 0.6, 0.95, 20
    else:
        final_prompt = f"/no_think {args.prompt}"
        rec_temp, rec_top_p, rec_top_k = 0.7, 0.8, 20

    # 2. Override with CLI args if they are explicitly provided by the user
    # We check if they are not None (using None as default in parser)
    temp = args.temperature if args.temperature is not None else rec_temp
    top_p = args.top_p if args.top_p is not None else rec_top_p
    top_k = args.top_k if args.top_k is not None else rec_top_k
    min_p = args.min_p # Always uses the default 0.0 unless user changes it

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-4B-AWQ",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=temp,
            top_p=top_p,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
            },
            max_tokens=args.max_tokens
        )
        
        output = response.choices[0].message.content
        
        print("\n" + "="*40)
        print(f"MODE:   {args.mode.upper()}")
        print(f"PARAMS: Temp={temp}, TopP={top_p}, TopK={top_k}, MinP={min_p}")
        print("-" * 40)
        print(f"RESPONSE:\n{output}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["think", "no_think"], default="no_think")
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    # Set default=None so we can detect if the user actually passed a value
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=0.0)
    
    args = parser.parse_args()
    run_single_inference(args)

