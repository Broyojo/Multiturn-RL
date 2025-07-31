import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from sandbox import Sandbox
from util import run_async

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MultiturnAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level MultiturnAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        rollout_config = messages[0]["content"]

        async with Sandbox(**rollout_config["sandbox"]) as sandbox:
            prompt_ids = await run_async(
                lambda: self.tokenizer.apply_chat_template(
                    messages[1:], tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
                )
            )
            response_mask = []

            user_turns, assistant_turns = 0, 0
            while True:
                with simple_timer("generate_sequences", metrics):
                    response_ids = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                    )
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                assistant_turns += 1

                # reach max response length
                if len(response_mask) >= self.response_length:
                    break

                # reach max assistant turns
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break

                # reach max user turns
                if self.max_user_turns and user_turns >= self.max_user_turns:
                    break

                # no tool calls
                _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                if not tool_calls:
                    break

                # call tools
                tasks = []
                for tool_call in tool_calls[: self.max_parallel_calls]:
                    tasks.append(self._call_tool(tool_call))
                with simple_timer("tool_calls", metrics):
                    tool_responses = await asyncio.gather(*tasks)
                if any(isinstance(item, Exception) for item in tool_responses):
                    break
