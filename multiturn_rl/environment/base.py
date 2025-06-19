from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    @abstractmethod
    async def rollout(
        self, prompt: list[dict[str, str]], server_address: str, sampling_params: dict[str, Any], config: dict[str, Any]
    ):
        pass
