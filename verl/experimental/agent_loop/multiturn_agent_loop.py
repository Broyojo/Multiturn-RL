import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from sandbox import Sandbox
from util import run_async

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("multiturn_agent")
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
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        rollout_config = json.loads(kwargs["raw_prompt"][0]["content"])

        async with Sandbox(**rollout_config["sandbox"]) as sandbox:
            response = await sandbox.session.list_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in response.tools
            ]
            prompt_ids = await run_async(
                lambda: self.tokenizer.apply_chat_template(
                    kwargs["raw_prompt"][1:], tools=available_tools, add_generation_prompt=True, tokenize=True
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
                    tasks.append(self._call_tool(tool_call, sandbox))
                with simple_timer("tool_calls", metrics):
                    tool_responses = await asyncio.gather(*tasks)
                if any(isinstance(item, Exception) for item in tool_responses):
                    break

                tool_response_ids = await run_async(
                    lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True
                    )
                )
                tool_response_ids = tool_response_ids[len(self.system_prompt) :]

                if len(response_mask) + len(tool_response_ids) >= self.response_length:
                    break

                prompt_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)
                user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, sandbox: Sandbox) -> dict[str, str]:
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool_response = await sandbox.session.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return {"role": "tool", "content": f"Error executing tool {tool_name}: {str(e)}"}

        tool_content = tool_response.content

        if len(tool_content) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_content = tool_content[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_content = "(truncated)..." + tool_content[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_content = tool_content[:length] + "...(truncated)..." + tool_content[-length:]

        return {
            "role": "tool",
            "content": tool_content,
        }
