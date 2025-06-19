from multiturn_rl.environment.base import BaseEnv


class EnvRegistry:
    _registry: dict[str, type[BaseEnv]] = {}

    def register(cls, name: str):
        def decorator(env_cls: type[BaseEnv]):
            cls._registry[name] = env_cls
            return env_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseEnv]:
        if not cls.is_registered(name):
            available = cls.list_available()
            raise KeyError(f"Environment '{name}' not found. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry


env_registry = EnvRegistry()
