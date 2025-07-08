# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import random
import time

import aiohttp
import numpy as np
import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict
from tqdm import tqdm

from verl.protocol import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer

logger = logging.getLogger(__file__)


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: list[str],
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: list[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        self.server_addresses = server_addresses
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers", {})
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.time()
        sampling_params = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = self.config.val_kwargs.top_p
            sampling_params["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {sampling_params}")

        group_size = 1 if batch.meta_info.get("validate", False) else self.config.n

        progress_bar = tqdm(desc="Rollout Progress", total=len(batch.non_tensor_batch["raw_prompt"]) * group_size)

        async def rollout(prompt: list[dict[str, str]], server_address: str, extra_info: dict, sampling_params: dict):
            # TODO: init sandbox here
            result = await self._chat_completions_aiohttp(server_address, messages=prompt, **sampling_params)
            message = result.choices[0].message
            progress_bar.update(1)
            return {"messages": prompt + [{"role": message.role, "content": message.content}], "tools": [], "reward": random.random()}

        tasks = []
        for i, (raw_prompt, extra_info) in enumerate(zip(batch.non_tensor_batch["raw_prompt"], batch.non_tensor_batch["extra_info"], strict=True)):
            server_address = self.server_addresses[i % len(self.server_addresses)]
            for _ in range(group_size):
                tasks.append(rollout(prompt=list(raw_prompt), server_address=server_address, extra_info=extra_info, sampling_params=sampling_params))

        results = await asyncio.gather(*tasks)
        progress_bar.close()

        output_batch = self._postprocess(
            batch=batch,
            batch_messages=[result["messages"] for result in results],
            tools=[result["tools"] for result in results],
            rewards=[result["reward"] for result in results],
            n=group_size,
        )

        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print("[ChatCompletionScheduler] generate_sequences done")

        return output_batch

    def _postprocess(self, batch: DataProto, batch_messages: list[list[dict[str, str]]], tools: list[dict], rewards: list[float], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, tools=tool_schemas, add_generation_prompt=True, tokenize=False) for prompt, tool_schemas in zip(batch.non_tensor_batch["raw_prompt"], tools)]
        assert len(batch_messages) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(messages, tools=tool_schema, add_generation_prompt=False, tokenize=False) for messages, tool_schema in zip(batch_messages, tools)]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        loss_mask = self._create_loss_mask(batch_messages, input_ids, attention_mask)
        response_mask = loss_mask[:, -responses["input_ids"].size(1) :]

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "loss_mask": loss_mask,  # [bsz, prompt_length + response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(messages) for messages in batch_messages], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns, "scores": np.array(rewards)})

    def _create_loss_mask(self, batch_messages, input_ids, attention_mask):
        batch_size = len(batch_messages)
        loss_mask = torch.zeros_like(attention_mask, dtype=torch.float)

        for i in range(batch_size):
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)

            turn_start = 0
            for j, msg in enumerate(batch_messages[i]):
                turn_end = eos_indices[j] if j < len(eos_indices) else len(input_ids[i]) - 1
                if msg["role"] == "assistant":
                    # note: beware of default system message in chat template, will mess up indexing
                    loss_mask[i, turn_start + 4 : turn_end + 1] = 1.0
                turn_start = turn_end + 1

        loss_mask = loss_mask * attention_mask

        return loss_mask
