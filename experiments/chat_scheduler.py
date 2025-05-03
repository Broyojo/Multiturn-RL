import asyncio
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from tensordict import TensorDict
from terminal import Terminal

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler


def make_trajectory(messages, extra_info):
    terminal = Terminal(
        image=extra_info["docker_image"], commit=extra_info["base_commit"]
    )
    return {"messages": list(messages), "terminal": terminal, "extra_info": extra_info}


GLOBAL_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="docker")


async def call_terminal(term: Terminal, input: str, timeout=1):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        GLOBAL_POOL, lambda: term(input=input, timeout=timeout)
    )


def extract_action(response):
    last_closing = response.rfind("</terminal>")
    last_opening = response.rfind("<terminal>")
    if last_closing > last_opening or last_opening == -1:
        return None
    return response[last_opening + 10 :]


def save_predictions(patches, n):
    """
    need to save several prediction files (N of them) since all instance ids must be unique.
    so, if we have GRPO, we generate n trajectories for each problem, so we need to have N predictinos for each instance
    maybe run the swesmith evaluator in parallel across each of the N prediction files
    """

    # Group patches by instance_id
    patches_by_instance = defaultdict(list)
    for patch in patches:
        instance_id = patch[KEY_INSTANCE_ID]
        patches_by_instance[instance_id].append(patch)

    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)

    # Stratify the predictions into old_n files
    for file_idx in range(n):
        predictions_for_file = []

        # Add one prediction per instance_id to this file
        for instance_id, instance_patches in patches_by_instance.items():
            if file_idx < len(instance_patches):
                predictions_for_file.append(instance_patches[file_idx])

        # Write this file's predictions
        output_path = f"predictions/predictions_{file_idx}.jsonl"
        with open(output_path, "w") as f:
            for patch in predictions_for_file:
                f.write(json.dumps(patch) + "\n")


class TerminalChatCompletionScheduler(ChatCompletionScheduler):
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: list[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)

    async def generate_sequences(
        self, batch: DataProto, **sampling_params
    ) -> DataProto:
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

        trajectories = Parallel(n_jobs=-1, backend="threading")(
            delayed(make_trajectory)(messages, extra_info)
            for messages, extra_info in zip(
                batch.non_tensor_batch["raw_prompt"],
                batch.non_tensor_batch["extra_info"],
            )
            for _ in range(kwargs["n"])
        )

        old_n = kwargs["n"]
        kwargs["n"] = 1
        kwargs["stop"] = ["</terminal>"]

        kwargs.update(sampling_params)
        print(
            f"[{self.__class__.__name__}] generate_sequences sampling params: {kwargs}"
        )

        async def callback(
            completions: ChatCompletion,
            info: dict[str, Any],
            exception: Exception | None,
        ):
            index, messages, terminal, batch_messages = (
                info["index"],
                info["messages"].copy(),
                info["terminal"],
                info["batch_messages"],
            )

            if exception is not None:
                # this may be from the terminal output overstepping the context length. in this case, we just return the messages but don't include the terminal output
                print(f"Callback exception: {exception}")
                batch_messages[index] = messages[:-1]
                return

            response = completions.choices[0]
            messages.append(
                {"role": response.message.role, "content": response.message.content}
            )

            if response.finish_reason == "length":
                batch_messages[index] = messages
                return

            action = extract_action(response.message.content)
            if action is None:
                batch_messages[index] = messages
                return

            messages[-1]["content"] += "</terminal>"

            output = await call_terminal(terminal, action, timeout=1)
            messages.append(
                {
                    "role": "user",
                    "content": f"<terminal_output>{output}</terminal_output>",
                }
            )
            print("*************** <terminal> call :", messages)
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

        def make_patch(t):
            patch = {
                KEY_INSTANCE_ID: t["extra_info"]["instance_id"],
                KEY_MODEL: self.model_name,
                KEY_PREDICTION: t["terminal"].get_patch(t["extra_info"]["base_commit"]),
            }
            t["terminal"].stop()
            return patch

        patches = Parallel(n_jobs=-1, backend="threading")(
            delayed(make_patch)(traj) for traj in trajectories
        )

        save_predictions(patches, old_n)

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
            self.tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]
        assert len(batch_messages) == len(prompts) * n

        sequences = [
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            for messages in batch_messages
        ]

        responses = [
            sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)
        ]

        prompts = self.tokenizer(
            prompts, return_tensors="pt", padding="longest", padding_side="left"
        )
        responses = self.tokenizer(
            responses, return_tensors="pt", padding="longest", padding_side="right"
        )
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(
                n, dim=0
            )

        # TODO: add assistant message masking here

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat(
            [prompts["attention_mask"], responses["attention_mask"]], dim=1
        )
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

    async def generate_sequences(
        self, batch: DataProto, **sampling_params
    ) -> DataProto:
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
        print(
            f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}"
        )

        async def callback(
            completions: ChatCompletion, info: Dict[str, Any], exception: Exception
        ):
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append(
                    {"role": choice.message.role, "content": choice.message.content}
                )
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(
            batch.non_tensor_batch["raw_prompt"]
        ):
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
        self,
        batch: DataProto,
        batch_conversations: List[List[List[Dict[str, str]]]],
        n: int,
    ) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [
            self.tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [
            conversation
            for conversations in batch_conversations
            for conversation in conversations
        ]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [
            self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=False, tokenize=False
            )
            for conversation in batch_conversations
        ]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [
            sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)
        ]

        prompts = self.tokenizer(
            prompts, return_tensors="pt", padding="longest", padding_side="left"
        )
        responses = self.tokenizer(
            responses, return_tensors="pt", padding="longest", padding_side="right"
        )
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(
                n, dim=0
            )

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat(
            [prompts["attention_mask"], responses["attention_mask"]], dim=1
        )
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
