#!/usr/bin/env python3
import argparse
from zorn import ZornChat

p = argparse.ArgumentParser(description="ZORN BitNet CLI")
p.add_argument("model_dir")
p.add_argument("--threads", type=int, default=None)
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--top-k", type=int, default=40)
p.add_argument("--top-p", type=float, default=0.9)
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--system", default=None)
a = p.parse_args()
chat = ZornChat(a.model_dir, cpu_cores=a.threads, system_prompt=a.system)
print("ZORN chat ready. Type /exit to quit, /reset to reset context.")
while True:
    try:
        q = input("\nYou: ")
    except (EOFError, KeyboardInterrupt):
        break
    if q.strip() == "/exit":
        break
    if q.strip() == "/reset":
        chat.reset(); print("[context reset]"); continue
    print("Assistant: ", end="", flush=True)
    for piece in chat.stream(q, max_new_tokens=a.max_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p):
        print(piece, end="", flush=True)
    print()
