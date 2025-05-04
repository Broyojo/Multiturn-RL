import re

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_assistant_mask(tokenizer: PreTrainedTokenizer, sequences: list[str]) -> torch.Tensor:
    encoding = tokenizer(
        sequences, return_tensors="pt", padding="longest", padding_side="right", return_offsets_mapping=True
    )
    masks = []
    for i, seq in enumerate(sequences):
        matches = re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", seq, re.S)
        mask = torch.zeros_like(encoding["input_ids"][i])
        for match in matches:
            pass

        masks.append(mask)

    return


batch_messages = [
    [
        {"role": "system", "content": "You are a helpful AI"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thanks! What can I do for you?"},
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did no one believe the atom? Because they make everything up!"},
    ],
    [
        {"role": "system", "content": "You are a helpful AI"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have real-time weather data, but I'd be happy to help you find it!"},
    ],
    [
        {"role": "system", "content": "You are a helpful AI"},
        {"role": "user", "content": "Explain quantum computing."},
        {
            "role": "assistant",
            "content": "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously.",
        },
        {"role": "user", "content": "That's interesting!"},
        {"role": "assistant", "content": "I'm glad you find it interesting! Would you like to know more about it?"},
    ],
]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    sequences = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        for messages in batch_messages
    ]

    mask = get_assistant_mask(tokenizer, sequences)
    print(mask)

# prompt = tokenizer.apply_chat_template(
#     [
#         {"role": "system", "content": "You are a helpful AI"},
#         {"role": "user", "content": "Hello, how are you?"},
#     ],
#     return_tensor="pt",
#     add_generation_prompt=True,
# )

# tokenized = tokenizer.apply_chat_template(messages, tokenize=False)

# encoding = tokenizer(tokenized, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
# offsets = encoding["offset_mapping"]

# # NOTE: <|im_end|> is just one extra token at the end
# matches = re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", tokenized, re.S)

# # Create mask for all assistant messages
# mask = torch.zeros_like(encoding["input_ids"])
# offset_list = offsets[0].tolist()  # Convert to list once for faster access

# for match in re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", tokenized, re.S):
#     content_start = match.start(1)  # Start of assistant content
#     content_end = match.end(1)  # End of assistant content
#     im_end_start = match.end(1)  # Start of <|im_end|> token
#     im_end_end = match.end(0)  # End of the entire match

#     # Find all tokens that fall within the assistant message (content + end token)
#     for i, (token_start, token_end) in enumerate(offset_list):
#         # Check if token is within the assistant content
#         if content_start <= token_start < content_end and token_start < token_end <= content_end:
#             mask[0][i] = 1
#         # Check if token is the <|im_end|> token
#         elif im_end_start <= token_start < im_end_end:
#             mask[0][i] = 1


# mask = torch.zeros_like(encoding["input_ids"])


# for match in matches:
#     content_start = match.start(1)
#     content_end = match.end(1)
#     print(content_start, content_end)

#     in_assistant_message = False
#     for i in range(len(mask[0])):
#         start, end = offsets[0][i]
#         if content_start <= start <= content_end and content_start <= end <= content_end:
#             mask[0][i] = 1
#             in_assistant_message = True
#         elif i > 0 and in_assistant_message:
#             mask[0][i] = 1  # account for <|im_end|> token
#             in_assistant_message = False

# print(mask)

# for i in range(len(mask[0])):
#     if mask[0, i] == 1:
#         print(f"[{tokenizer.decode(encoding['input_ids'][0, i].item())}]", end=" ")
# print()

# print(mask[0, len(prompt) :])

# print(tokenized)
# encoding = tokenizer(
#     tokenized, add_special_tokens=False, truncation=True, return_offsets_mapping=True, return_tensors="pt"
# )
# all_offsets = encoding["offset_mapping"]
# print(all_offsets)

# msgs_batch = [
#     [  # Batch 1
#         {"role":"user", "content":"Hello, how are you?"},
#         {"role":"assistant", "content":"I'm fine, thanks! What can I do for you?"},
#         {"role":"user", "content":"Tell me a joke."},
#         {"role":"assistant", "content":"Why did no one believe the atom? Because they make everything up!"},
#     ],
#     [  # Batch 2
#         {"role":"user", "content":"What's the weather like?"},
#         {"role":"assistant", "content":"I don't have real-time weather data, but I'd be happy to help you find it!"},
#     ],
#     [  # Batch 3
#         {"role":"user", "content":"Explain quantum computing."},
#         {"role":"assistant", "content":"Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously."},
#         {"role":"user", "content":"That's interesting!"},
#         {"role":"assistant", "content":"I'm glad you find it interesting! Would you like to know more about it?"},
#     ]
# ]


# def assistant_mask(batch_messages):
#     texts = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=False)
#     # text = text[:-len("<|im_end|>")-1]
#     encoding = tokenizer(texts, add_special_tokens=False, truncation=True, return_offsets_mapping=True, return_tensors="pt")
#     all_offsets = encoding["offset_mapping"]

#     pattern = re.escape("<|im_start|>assistant\n") + r"(.*?(?:" + re.escape("<|im_end|>") + r"|$))"
#     print(pattern)
#     assistant_start_end = [[(m.start(1), m.end(1)) for m in re.finditer(pattern, text, re.DOTALL)] for text in texts]

#     mask = torch.zeros_like(encoding.input_ids)

#     for i, offsets in enumerate(all_offsets):
#         for j, offset in enumerate(offsets):
#             offset_start, offset_end = offset[0].item(), offset[1].item()
#             token_in_assistant = False

#             for start, end in assistant_start_end[i]:
#                 if start <= offset_start and offset_end <= end:
#                     token_in_assistant = True
#                     break

#             mask[i, j] = 1 if token_in_assistant else 0

#     return mask

# mask = assistant_mask(msgs_batch)

# text = tokenizer.apply_chat_template(msgs_batch, tokenize=False, add_generation_prompt=False)
# encoding = tokenizer(text, add_special_tokens=False, truncation=True, return_offsets_mapping=True, return_tensors="pt")
# masked = mask * encoding.input_ids

# print(masked)

# print(tokenizer.batch_decode(masked))
