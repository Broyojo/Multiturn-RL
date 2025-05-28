import asyncio
import logging
import re
import resource
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from reward import swebench_eval, swesmith_eval
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from tensordict import TensorDict
from terminal import Terminal
from tqdm import tqdm

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler

resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))

logging.getLogger("httpx").setLevel(logging.WARNING)


class TerminalChatCompletionScheduler(ChatCompletionScheduler):
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: list[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)
        self.train_step = 1
        self.eval_step = 1
        self.thread_pool = ThreadPoolExecutor(max_workers=128, thread_name_prefix="chatcompletions")

    async def _run_in_async(self, func):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, func)

    def _extract_action(self, response):
        last_closing = response.rfind("</terminal>")
        last_opening = response.rfind("<terminal>")
        if last_closing > last_opening or last_opening == -1:
            return None
        return response[last_opening + 10 :]

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

        trajectories = [
            {
                "messages": list(messages),
                "terminal": None,
                "patch": "",
                "extra_info": extra_info,
                "turn": 0,
            }
            for messages, extra_info in zip(
                batch.non_tensor_batch["raw_prompt"],
                batch.non_tensor_batch["extra_info"],
                strict=False,
            )
            for _ in range(kwargs["n"])
        ]

        old_n = kwargs["n"]
        kwargs["n"] = 1
        kwargs["stop"] = ["</terminal>"]

        kwargs.update(sampling_params)
        print(f"[{self.__class__.__name__}] generate_sequences sampling params: {kwargs}")

        bar = tqdm(total=len(trajectories), desc="Rollout Progress")

        async def callback(completions: ChatCompletion, info: dict[str, Any], exception: Exception | None):
            traj = info["traj"]
            messages, terminal, extra_info = traj["messages"], traj["terminal"], traj["extra_info"]

            def cleanup():
                patch = {
                    KEY_INSTANCE_ID: extra_info["instance_id"],
                    KEY_MODEL: self.model_name,
                    KEY_PREDICTION: terminal.get_patch(extra_info["base_commit"]) if terminal is not None else "",
                }

                if terminal is not None:
                    terminal.stop()
                    source = traj["extra_info"]["source"]
                    data_row = traj["extra_info"]["data_row"]
                    if source == "swesmith":
                        traj["score"] = ray.get(swesmith_eval.remote(patch, data_row, self.train_step))
                    elif source == "swebench":
                        traj["score"] = ray.get(swebench_eval.remote(patch, data_row, self.eval_step))
                    else:
                        raise ValueError(f"unknown data source: {source}")
                else:
                    traj["score"] = 0
                bar.update(1)

            if exception is not None:
                traj["finish_reason"] = "exception"
                messages.pop()
                await self._run_in_async(cleanup)
                return

            response = completions.choices[0]
            messages.append({"role": response.message.role, "content": response.message.content})

            if response.finish_reason == "length":
                traj["finish_reason"] = "length"
                await self._run_in_async(cleanup)
                return

            action = self._extract_action(response.message.content)
            if action is None:
                traj["finish_reason"] = "stop"
                await self._run_in_async(cleanup)
                return

            messages[-1]["content"] += "</terminal>"

            traj["turn"] += 1
            if traj["turn"] > self.config.max_turns:
                traj["finish_reason"] = "max_turns"
                await self._run_in_async(cleanup)
                return

            if terminal is None:
                traj["terminal"] = await self._run_in_async(
                    lambda: Terminal(image=extra_info["docker_image"], commit=extra_info["base_commit"])
                )
                terminal = traj["terminal"]

            output = await self._run_in_async(lambda: terminal(action, timeout=1))
            messages.append(
                {
                    "role": "user",
                    "content": f"<terminal_output>{output}</terminal_output>",
                }
            )

            if await self._run_in_async(
                lambda: len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
                > self.config.max_model_len - self.config.response_length
            ):
                traj["finish_reason"] = "overflow"
                messages.pop()
                await self._run_in_async(cleanup)
                return

            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={"traj": traj},
                model=self.model_name,
                messages=messages,
                **kwargs,
            )

        tasks = []
        for traj in trajectories:
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={"traj": traj},
                        model=self.model_name,
                        messages=traj["messages"],
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print(f"[{self.__class__.__name__}] generate_sequences done")

        messages = [traj["messages"] for traj in trajectories]
        scores = [traj["score"] for traj in trajectories]

        if is_validate:
            self.eval_step += 1
        else:
            self.train_step += 1

        return self._postprocess(batch=batch, batch_messages=messages, scores=scores, n=old_n)

    def _get_assistant_mask(self, responses: list[str]) -> torch.Tensor:
        responses = ["<|im_start|>assistant\n" + response for response in responses]

        encoding = self.tokenizer(
            responses, return_tensors="pt", padding="longest", padding_side="right", return_offsets_mapping=True
        )
        masks = []
        for i, response in enumerate(responses):
            matches = re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", response, re.S)
            mask = torch.zeros_like(encoding["input_ids"][i])
            offset_list = encoding["offset_mapping"][i].tolist()
            for match in matches:
                content_start = match.start(1)
                content_end = match.end(1)
                im_end_start = match.end(1)
                im_end_end = match.end(0)

                for i, (token_start, token_end) in enumerate(offset_list):
                    if content_start <= token_start < content_end and token_start < token_end <= content_end:
                        mask[i] = 1
                    elif im_end_start <= token_start < im_end_end:
                        mask[i] = 1

            masks.append(mask[3:])

        return torch.stack(masks)

    def _postprocess(
        self,
        batch: DataProto,
        batch_messages: list[list[dict[str, str]]],
        scores: list[float],
        n: int,
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
        assistant_mask = self._get_assistant_mask(responses)

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
                "assistant_mask": assistant_mask,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch, non_tensor_batch={"scores": np.array(scores)})
