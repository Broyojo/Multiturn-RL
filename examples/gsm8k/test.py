import ray

from multiturn_rl.env.math_env import MathEnv
from multiturn_rl.env.registry import EnvRegistry

MathEnv

math_env = ray.remote(EnvRegistry.get("MathEnv")).remote()

print(math_env)
math_env.rollout.remote(prompt=[{"role": "user", "content": "what is 2+2?"}])
