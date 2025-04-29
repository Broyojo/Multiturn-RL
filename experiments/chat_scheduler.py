import asyncio
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict
from terminal import Terminal
from tqdm import tqdm

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler


def create_terminal(messages, docker_image):
    terminal = Terminal(image=docker_image).__enter__()
    return {
        "messages": list(messages),
        "terminal": terminal
    }

def create_terminals(batch, n_samples):
    terminal_args = []
    for messages, extra_info in zip(batch.non_tensor_batch["raw_prompt"], 
                                    batch.non_tensor_batch["extra_info"]):
        for _ in range(n_samples):
            terminal_args.append((messages, extra_info["docker_image"]))
    
    terminals = []
    max_workers = min(os.cpu_count(), 24)
    
    with tqdm(total=len(terminal_args), desc="Creating terminals") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(create_terminal, *args): args for args in terminal_args}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    terminals.append(result)
                except Exception as e:
                    print(f"Failed to create terminal: {e}")
                    traceback.print_exc()
                
                pbar.update(1)
    
    return terminals

GLOBAL_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="docker")

async def stop_terminal(term: Terminal, timeout: int = 1):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(GLOBAL_POOL, lambda: term.stop(timeout=timeout))

async def call_terminal(term: Terminal, input: str, timeout=1):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(GLOBAL_POOL, lambda: term(input=input, timeout=timeout))

def extract_action(response):
    last_closing = response.rfind("</terminal>")
    last_opening = response.rfind("<terminal>")
    if last_closing > last_opening or last_opening == -1:
        return None
    return response[last_opening + 10:]

class TerminalChatCompletionScheduler(ChatCompletionScheduler):
    def __init__(self, config: DictConfig, model_path: str, server_addresses: list[str], max_cache_size: int = 10000):
        super().__init__(config, model_path, server_addresses, max_cache_size)
    
    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        trajectories = create_terminals(batch, kwargs["n"])
        old_n = kwargs["n"]
        kwargs["n"] = 1
        kwargs["stop"] = ["</terminal>"]

        kwargs.update(sampling_params)
        print(f"[{self.__class__.__name__}] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: dict[str, Any], exception: Exception | None):
            index, messages, terminal, batch_messages = (
                info["index"],
                info["messages"].copy(),
                info["terminal"],
                info["batch_messages"]
            )
            
            if exception is not None:
                print(f"Callback exception: {exception}")
                batch_messages[index] = messages
                await stop_terminal(terminal)
                return

            response = completions.choices[0]
            messages.append({
                "role": response.message.role,
                "content": response.message.content
            })
            
            if response.finish_reason == "length":
                batch_messages[index] = messages
                await stop_terminal(terminal)
                # print("finish_reason=='length' :", messages)
                return
            
            action = extract_action(response.message.content)
            if action is None:
                batch_messages[index] = messages
                await stop_terminal(terminal)
                # print("<|im_end|> :", messages)
                return
            
            print("*************** </terminal> :", messages)
            messages.append({
                "role": "user",
                "content": f"<output>{await call_terminal(terminal, action, timeout=1)}</output>"
            })
            print("*************** response :", messages[-1])
            # TODO: is there a way to detect that we don't overstep the model context len? maybe we look at the exception arg
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "index": index,
                    "messages": messages,
                    "terminal": terminal,
                    "batch_messages": batch_messages,
                },
                model=self.model_name,
                messages=messages,
                **kwargs,
            )

        tasks = []
        batch_messages = [None] * len(trajectories)
        for i, traj in enumerate(trajectories):
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "index": i,
                            "messages": traj["messages"],
                            "terminal": traj["terminal"],
                            "batch_messages": batch_messages,
                        },
                        model=self.model_name,
                        messages=traj["messages"],
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print(f"[{self.__class__.__name__}] generate_sequences done")
        
        return self._postprocess(batch, batch_messages, old_n)

    def _postprocess(
        self, batch: DataProto, batch_messages: list[list[dict[str, str]]], n: int
    ) -> DataProto:
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        
        prompts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            for prompt in batch.non_tensor_batch["raw_prompt"] 
        ]
        assert len(batch_messages) == len(prompts) * n
        
        sequences = [
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            for messages in batch_messages
        ]
        
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]
        
        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)
        
        # TODO: add assistant message masking here

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)




class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                        },
                        model=self.model_name,
                        messages=conversation,
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])

    def _postprocess(
        self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int
    ) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [
            self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
            for conversation in batch_conversations
        ]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)
