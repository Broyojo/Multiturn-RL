import asyncio
import time

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from multiturn_rl.env.registry import EnvRegistry
from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler


class MultiturnChatCompletionScheduler(ChatCompletionScheduler):
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: list[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        t_start = time.time()
        sampling_params = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            model=self.model_name,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            sampling_params["n"] = 1
            sampling_params["temperature"] = 0

        group_size = sampling_params["n"]
        sampling_params["n"] = 1

        server_addresses = [address[1] for address in self.weighted_addresses]
        envs = {}
        tasks = []

        prompts = batch.non_tensor_batch["raw_prompt"]
        extra_infos = batch.non_tensor_batch["extra_info"]

        for i, (prompt, extra_info) in enumerate(zip(prompts, extra_infos, strict=True)):
            env_name = extra_info["env"]["name"]
            if env_name not in envs:
                envs[env_name] = ray.remote(EnvRegistry.get(env_name)).remote()

            server_addr = server_addresses[i % len(server_addresses)]

            for _ in range(group_size):
                task = envs[env_name].rollout.remote(
                    prompt=prompt,
                    server_address=server_addr,
                    sampling_params=sampling_params,
                    config=extra_info["env"]["rollout_config"],
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        output_batch = self._postprocess(
            batch=batch,
            batch_messages=[result["messages"] for result in results],
            batch_tools=[result.get("tools", None) for result in results],
            batch_rewards=[result["reward"] for result in results],
            n=group_size,
        )
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print(f"[{self.__name__}] generate_sequences done")
        return output_batch

    def _postprocess(
        self,
        batch: DataProto,
        batch_messages: list[list[dict]],
        batch_tools: list[list[dict | None]],
        batch_rewards: list[list[float]],
        n: int,
    ) -> DataProto:
        tokenizer = self.completion_callback.tokenizer
        prompts = [
            tokenizer.apply_chat_template(prompt, tools=tools, add_generation_prompt=True, tokenize=False)
            for prompt, tools in zip(batch.non_tensor_batch["raw_prompt"], batch_tools, strict=True)
        ]
        assert len(batch_messages) == len(prompts) * n

        sequences = [
            tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False)
            for messages, tools in zip(batch_messages, batch_tools, strict=False)
        ]

        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        loss_mask = self._compute_loss_mask(
            batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0),
            batch_messages,
            responses["input_ids"],
            responses["attention_mask"],
        )

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "loss_mask": loss_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(messages) for messages in batch_messages], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns, "scores": np.array(batch_rewards)})

    def _compute_loss_mask(
        self,
        raw_prompts: list[list[dict[str, str]]],
        batch_messages: list[list[dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a mask that is 1 only over the assistant's tokens,
        and 0 everywhere else (user, tool, padding, etc.).
        """
        batch_size = input_ids.size(0)
        loss_mask = torch.zeros_like(attention_mask)

        for i in range(batch_size):
            # get just the response messages for this example
            responses = batch_messages[i][len(raw_prompts[i]) :]
            # roles in order of messages
            roles = [msg["role"] for msg in responses]
            # find the positions of each EOS token up to number of messages
            eos_positions = (input_ids[i].eq(self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0].tolist())[
                : len(roles)
            ]

            for j, role in enumerate(roles):
                # compute begin-of-segment (just after previous EOS, or 0)
                bos = eos_positions[j - 1] + 1 if j > 0 else 0
                eos = eos_positions[j]
                if role == "assistant":
                    # keep only assistant spans
                    loss_mask[i, bos : eos + 1] = 1

        return loss_mask
