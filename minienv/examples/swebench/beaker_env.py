import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from minienv.backend.beaker import BeakerBackend


@dataclass
class BeakerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""


class BeakerEnvironment:
    """A basic wrapper around minienv"""
    def __init__(self, *, config_class: type = BeakerEnvironmentConfig, **kwargs):
        self.backend: Optional[BeakerBackend] = None
        self.config: BeakerEnvironmentConfig = config_class(**kwargs)
        self._start_container()

    def _start_container(self):
        env_vars = self.config.env
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None and key not in self.config.env:
                env_vars[key] = value

        self.backend = BeakerBackend(workspace="ai2/rollouts")

        asyncio.run(self.backend.create_env(
            task_name="minisweagent", 
            image=self.config.image,
            env_vars=env_vars,
        ))

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        cwd = cwd or self.config.cwd
        assert self.backend, "Container not started"

        stdout, stderr, returncode = asyncio.run(self.backend.exec_command(
            command=command, 
            timeout=self.config.timeout,
            cwd=cwd
        ))

        return {"output": stdout, "returncode": returncode}

    def cleanup(self):
        if self.backend is None:
            return

        asyncio.run(self.backend.teardown())

    def __del__(self):
        self.cleanup()
