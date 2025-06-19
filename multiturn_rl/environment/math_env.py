from gguf import Any
from math_verify import parse, verify
from openai import AsyncOpenAI

from multiturn_rl.environment.base import BaseEnv
from multiturn_rl.environment.registry import env_registry


@env_registry.register("math-v0")
class MathEnv(BaseEnv):
    async def rollout(
        self, prompt: list[dict[str, str]], server_address: str, sampling_params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        client = AsyncOpenAI(base_url=server_address)

        messages = []
        messages.extend(prompt)
        result = await client.chat.completions.create(messages=messages, **sampling_params)
        message = result.choices[0].message
        messages.append(message)

        # TODO: make these async or ray tasks
        gold = parse(config["ground_truth"])
        answer = parse(messages[-1]["content"])
        score = 1 if verify(gold, answer) else 0

        return {"messages": messages, "tools": [], "reward": score}


# class SWEEnv(BaseEnv):
#     async def rollout(self, messages, config, server_address, **sampling_params):
#         ip_addr = await launch_sandbox(config["image"])
#         mcp_client = await connect_mcp_client(ip_addr, port=8000)

#         while len(messages) < 100:
#             response = llm_generate(messages)
#             if response == "tool":
#                 tool_response = mcp_client.call_tool("tool", args)
#             messages.append(response)

#         reward = compute_reward(messages)

#         return messages, reward


# @mcp.call_tool()
# def call_tool(name, args):
#     pass


# @mcp.list_tool()
# def list_tool():
#     pass


# note to self:
# environment should just handle the high level logic, don't need to dive into tokenization or masking
# also, reward for each message is good enough since it is natural, not within a message
# completion callback handles the final tokenization before handing off to the trainer. we save rewards to rm_scores
# just base off the pre-existing completion callback for the tools
# all env needs to do is give the messages, tools, and rewards for the sequence so that it can be properly tokenized
# TODO: what if tools is very long, overstepping max_prompt_length??

# TODO: load balance the environments. maybe we have 50 rollouts per environment, so we spread out the work.
# going to 1 rollout per envionrment just degenerates to the 1 env per rollout case
# use an actor pool, basically while forming all the envs, just count up until the concurrency per env is reached
