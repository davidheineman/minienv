from abc import ABC, abstractmethod
from typing import Tuple


class Backend(ABC):
    """Abstract base class for execution backends."""
    
    @abstractmethod
    async def create_env(self, task_name: str, image: str, **kwargs) -> None:
        """Create and initialize the environment."""
        pass
    
    @abstractmethod
    async def exec_command(self, command: str, timeout: int = 60) -> Tuple[str, str, int]:
        """Execute a command and return (stdout, stderr, exit_code)."""
        pass
    
    @abstractmethod
    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the environment."""
        pass
    
    @abstractmethod
    async def download_file(self, source: str) -> bytes:
        """Download file content from the environment."""
        pass
    
    @abstractmethod
    async def teardown(self) -> bool:
        """Clean up the environment."""
        pass