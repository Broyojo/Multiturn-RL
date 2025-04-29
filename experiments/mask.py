import re

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

msgs_batch = [
    [  # Batch 1
        {"role":"user", "content":"Hello, how are you?"},
        {"role":"assistant", "content":"I'm fine, thanks! What can I do for you?"},
        {"role":"user", "content":"Tell me a joke."},
        {"role":"assistant", "content":"Why did no one believe the atom? Because they make everything up!"},
    ],
    [  # Batch 2
        {"role":"user", "content":"What's the weather like?"},
        {"role":"assistant", "content":"I don't have real-time weather data, but I'd be happy to help you find it!"},
    ],
    [  # Batch 3
        {"role":"user", "content":"Explain quantum computing."},
        {"role":"assistant", "content":"Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously."},
        {"role":"user", "content":"That's interesting!"},
        {"role":"assistant", "content":"I'm glad you find it interesting! Would you like to know more about it?"},
    ]
]


def assistant_mask(batch_messages):
    texts = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=False)
    # text = text[:-len("<|im_end|>")-1]
    encoding = tokenizer(texts, add_special_tokens=False, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    all_offsets = encoding["offset_mapping"]
    
    pattern = re.escape("<|im_start|>assistant\n") + r"(.*?(?:" + re.escape("<|im_end|>") + r"|$))"
    print(pattern)
    assistant_start_end = [[(m.start(1), m.end(1)) for m in re.finditer(pattern, text, re.DOTALL)] for text in texts]
    
    mask = torch.zeros_like(encoding.input_ids)
    
    for i, offsets in enumerate(all_offsets):
        for j, offset in enumerate(offsets):
            offset_start, offset_end = offset[0].item(), offset[1].item()
            token_in_assistant = False
            
            for start, end in assistant_start_end[i]:
                if start <= offset_start and offset_end <= end:
                    token_in_assistant = True
                    break
            
            mask[i, j] = 1 if token_in_assistant else 0
    
    return mask

mask = assistant_mask(msgs_batch)

text = tokenizer.apply_chat_template(msgs_batch, tokenize=False, add_generation_prompt=False)
encoding = tokenizer(text, add_special_tokens=False, truncation=True, return_offsets_mapping=True, return_tensors="pt")
masked = mask * encoding.input_ids

print(masked)

print(tokenizer.batch_decode(masked))