from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
messages = [
    {
        "role": "config",
        "content": {
            "sandbox": {
                "image": "timemagic/rl-mcp:general",
                "command": "/mcp/daemon-mcp.py",
                "user": "root",
                "working_dir": "/",
                "network": "mcp-network",
                "ports": {"3000/tcp": None},
                "mcp_port": 3000,
            }
        },
    }
]

print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
