import asyncio
import json
import logging
import os
import random
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import docker
from docker.models.containers import Container
from pydantic import BaseModel, Field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box

console = Console()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Rich formatting functions
def print_docker(message: str, status: str = "info") -> None:
    """Print Docker-related messages in blue."""
    
    color = {
        "info": "blue",
        "success": "green", 
        "error": "red",
        "warning": "yellow"
    }.get(status, "blue")
    
    console.print(f"🐳 {message}", style=f"bold {color}")

def print_model(message: str, role: str = "assistant") -> None:
    """Print model I/O messages in purple/magenta."""
    color = {
        "assistant": "bright_magenta",
        "user": "cyan", 
        "system": "dim white",
        "tool": "green"
    }.get(role, "bright_magenta")
    
    console.print(f"{message}", style=f"bold {color}")

def print_model_input(messages: List[Any], tools: List[str]) -> None:
    """Print the exact input being sent to the LLM."""
    # Get the actual conversation content
    content_lines = []
    content_lines.append(f"[bold yellow]Available Tools:[/] {', '.join(tools)}")
    content_lines.append("")
    
    # Show all messages in the conversation
    for i, msg in enumerate(messages):
        if hasattr(msg, 'role'):
            role_color = {
                'system': 'dim white',
                'user': 'cyan', 
                'assistant': 'bright_magenta',
                'tool': 'green'
            }.get(msg.role, 'white')
            
            content_lines.append(f"[bold {role_color}]Message {i+1} ({msg.role}):[/]")
            # Truncate very long messages but show enough context
            msg_content = msg.content if hasattr(msg, 'content') else str(msg)
            if len(msg_content) > 500:
                msg_content = msg_content[:500] + "... [truncated]"
            content_lines.append(msg_content)
            
            # Show tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                content_lines.append(f"[dim yellow]Tool calls: {len(msg.tool_calls)}[/]")
                for tc in msg.tool_calls:
                    args_preview = str(tc.arguments)[:100] + ("..." if len(str(tc.arguments)) > 100 else "")
                    content_lines.append(f"  {tc.function}({args_preview})")
            
            content_lines.append("")
    
    console.print(Panel(
        "\n".join(content_lines),
        title="MODEL CONTEXT",
        border_style="bright_blue",
        box=box.DOUBLE,
        width=120
    ))

def print_model_output(response_content: str, tool_calls: List[Any]) -> None:
    """Print the exact output from the LLM."""
    content_lines = []
    
    if response_content and response_content.strip():
        content_lines.append("[bold bright_magenta]MODEL REASONING:[/]")
        content_lines.append(response_content.strip())
        content_lines.append("")
    
    if tool_calls:
        content_lines.append(f"[bold yellow]TOOL CALLS REQUESTED ({len(tool_calls)}):[/]")
        for i, tc in enumerate(tool_calls, 1):
            # Show full arguments
            args_str = json.dumps(tc.arguments, indent=2) if hasattr(tc, 'arguments') else str(tc.arguments)
            content_lines.append(f"[cyan]{i}. {tc.function}[/]")
            content_lines.append(f"[dim]{args_str}[/]")
            if i < len(tool_calls):
                content_lines.append("")
    
    if not content_lines:
        content_lines = ["[dim]No response content or tool calls[/]"]
    
    console.print(Panel(
        "\n".join(content_lines),
        title="MODEL RESPONSE", 
        border_style="bright_magenta",
        box=box.DOUBLE,
        width=120
    ))

def print_container_action(action_type: str, details: str, result: str = "", success: bool = True) -> None:
    content_lines = []
    
    # Action header
    action_color = "green" if success else "red"
    status_symbol = "✅" if success else "❌"
    content_lines.append(f"[bold {action_color}]{status_symbol} ACTION:[/] {action_type}")
    content_lines.append(f"[bold cyan]DETAILS:[/] {details}")
    
    if result and result.strip():
        content_lines.append("")
        content_lines.append("[bold yellow]CONTAINER OUTPUT:[/]")
        # Limit output length but show key information
        if len(result) > 1000:
            lines = result.split('\n')
            if len(lines) > 20:
                shown_result = '\n'.join(lines[:10]) + f"\n... [{len(lines)-20} lines omitted] ...\n" + '\n'.join(lines[-10:])
            else:
                shown_result = result[:1000] + "... [truncated]"
        else:
            shown_result = result
        content_lines.append(f"[dim white]{shown_result}[/]")
    
    border_style = "green" if success else "red"
    console.print(Panel(
        "\n".join(content_lines),
        title="CONTAINER ACTION" if success else "CONTAINER ERROR",
        border_style=border_style,
        box=box.DOUBLE,
        width=120
    ))

def print_tool(tool_name: str, args: Dict[str, Any], status: str = "call") -> None:
    """Print tool-related messages in green/red."""
    if status == "call":
        console.print(f"  [bold yellow]Tool Call:[/] [cyan]{tool_name}[/]", end="")
        if args:
            # Format args nicely
            args_str = ", ".join([f"{k}='{str(v)[:50]}{'...' if len(str(v)) > 50 else ''}'" for k, v in args.items()])
            console.print(f"({args_str})", style="dim")
        else:
            console.print("()", style="dim")
    elif status == "success":
        console.print(f"✅ [bold green]Tool Result[/] ([cyan]{tool_name}[/]):", style="bold")
    elif status == "error":
        console.print(f"❌ [bold red]Tool Error[/] ([cyan]{tool_name}[/]):", style="bold")

def print_tool_result(content: str, success: bool = True, max_length: int = 300) -> None:
    """Print tool results with syntax highlighting but no panels."""
    # This function is now replaced by print_container_action, but keeping for compatibility
    pass

def print_progress(message: str) -> None:
    """Print progress messages in yellow."""
    console.print(f"[bold yellow]Progress:[/] {message}")

def print_turn_header(turn: int, max_turns: int) -> None:
    """Print turn header with progress bar."""
    # Create a simple progress indicator
    progress_bar = "█" * min(turn, 20) + "░" * max(0, 20 - turn)
    
    # Add clear separator
    console.print("\n" + "="*120, style="dim white")
    console.print(f"[bold blue]TURN {turn}/{max_turns}[/] [{progress_bar}]", style="bold blue")
    console.print("="*120, style="dim white")

def print_task_header(task_id: str, tools: List[str], time_limit: int, max_turns: int) -> None:
    """Print beautiful task header."""
    # Create a beautiful header panel
    header_content = f"""
[bold green]Task:[/] {task_id}
[bold blue]Available Tools:[/] {', '.join(tools)}
[bold yellow]Time Limit:[/] {time_limit}s
[bold cyan]Max Turns:[/] {max_turns}
    """
    
    console.print(Panel(
        header_content.strip(),
        title="Agent Config",
        border_style="bright_blue",
        box=box.ROUNDED
    ))

def print_final_result(success: bool, score: float, execution_time: float, output: str) -> None:
    """Print final results with rich formatting."""
    # Create results table
    table = Table(title="🏁 Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold")
    
    success_style = "bold green" if success else "bold red"
    success_text = "✅ Success" if success else "❌ Failed"
    
    table.add_row("Status", f"[{success_style}]{success_text}[/]")
    table.add_row("Score", f"[bold yellow]{score:.1f}[/]")
    table.add_row("Execution Time", f"[bold blue]{execution_time:.2f}s[/]")
    
    console.print(table)
    
    if output:
        console.print(Panel(
            output,
            title="Output",
            border_style="green" if success else "red"
        ))

# =============================================================================
# CORE ABSTRACTIONS
# =============================================================================

class NetworkMode(StrEnum):
    """Network configuration for containers."""
    NONE = auto()
    WEBCACHE_GET_ONLY = auto() 
    UNPROXIED = auto()
    API_ACCESS = auto()


class ExecutionResult(BaseModel):
    """Result of executing a shell command."""
    output: bytes
    exit_code: int
    
    @property
    def text_output(self) -> str:
        return self.output.decode('utf-8', errors='replace')


class JupyterExecutionResult(BaseModel):
    """Result of executing code in Jupyter kernel."""
    status: str
    output: str
    final_expression_output: Optional[str] = None
    exception: Optional[Dict[str, Any]] = None


@dataclass
class ContainerConfig:
    """Configuration for a Docker container."""
    image: str
    environment: Dict[str, str] = None
    volumes: Dict[str, str] = None  # host_path -> container_path
    ports: List[int] = None
    privileged: bool = False
    gpu_access: bool = False
    network_mode: NetworkMode = NetworkMode.UNPROXIED
    memory_limit: Optional[str] = None
    timeout: int = 3600  # seconds
    export_results: bool = True  # Export /results to host
    results_host_path: str = "./minienv_results"  # Where to save results on host (relative to current dir)
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.volumes is None:
            self.volumes = {}
        if self.ports is None:
            self.ports = []


# =============================================================================
# CONTAINER INTERFACE ABSTRACTION
# =============================================================================

class ComputerInterface(ABC):
    """Abstract interface for interacting with containerized environments."""
    
    @abstractmethod
    async def execute_shell(self, command: str, timeout: int = 60) -> ExecutionResult:
        """Execute a shell command in the container."""
        pass
    
    @abstractmethod
    async def execute_python(self, code: str, timeout: int = 60) -> JupyterExecutionResult:
        """Execute Python code in the container's Jupyter kernel."""
        pass
    
    @abstractmethod
    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the container."""
        pass
    
    @abstractmethod
    async def download_file(self, source: str) -> bytes:
        """Download file content from the container."""
        pass
    
    @abstractmethod
    async def disable_internet(self) -> None:
        """Disable internet access for the container."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up container resources."""
        pass


# =============================================================================
# DOCKER CONTAINER ORCHESTRATION
# =============================================================================

class PortManager:
    """Manages dynamic port allocation to avoid conflicts."""
    
    def __init__(self, port_range: Tuple[int, int] = (10000, 32767)):
        self.available_ports = set(range(port_range[0], port_range[1] + 1))
        self.allocated_ports = set()
    
    def allocate_port(self) -> int:
        """Allocate a random available port."""
        if not self.available_ports:
            raise RuntimeError("No available ports")
        
        port = random.choice(list(self.available_ports))
        self.available_ports.remove(port)
        self.allocated_ports.add(port)
        logger.info(f"Allocated port {port}")
        return port
    
    def release_port(self, port: int) -> None:
        """Release a previously allocated port."""
        if port in self.allocated_ports:
            self.allocated_ports.remove(port)
            self.available_ports.add(port)
            logger.info(f"Released port {port}")


class DockerContainer:
    """Manages a single Docker container's lifecycle."""
    
    def __init__(self, config: ContainerConfig, port_manager: PortManager):
        self.config = config
        self.port_manager = port_manager
        self.container: Optional[Container] = None
        self.host_ports: List[int] = []
        self.docker_client = docker.from_env()
        self.container_id = f"nanoeval-{uuid.uuid4().hex[:8]}"
    
    async def start(self) -> Container:
        """Start the Docker container with the given configuration."""
        try:
            # Pull image if needed
            print_docker(f"Pulling image: {self.config.image}", "info")
            await asyncio.to_thread(self.docker_client.images.pull, self.config.image)
            
            # Allocate ports
            self.host_ports = [
                self.port_manager.allocate_port() 
                for _ in self.config.ports
            ]
            
            # Configure container options
            container_options = {
                'image': self.config.image,
                'name': self.container_id,
                'detach': True,
                'environment': self.config.environment,
                'remove': False,  # We'll remove manually for cleanup control
                'command': ['sh', '-c', 'mkdir -p /results && sleep infinity'],  # Create /results and keep container running
            }
            
            # Port mapping
            if self.config.ports:
                container_options['ports'] = {
                    f"{container_port}/tcp": host_port
                    for container_port, host_port in zip(self.config.ports, self.host_ports)
                }
            
            # Volume mounts
            if self.config.volumes:
                container_options['volumes'] = {
                    host_path: {'bind': container_path, 'mode': 'rw'}
                    for host_path, container_path in self.config.volumes.items()
                }
            
            # GPU access
            if self.config.gpu_access:
                container_options['runtime'] = 'nvidia'
                container_options['environment']['NVIDIA_VISIBLE_DEVICES'] = 'all'
            
            # Privileged mode
            if self.config.privileged:
                container_options['privileged'] = True
            
            # Memory limit
            if self.config.memory_limit:
                container_options['mem_limit'] = self.config.memory_limit
            
            # Network configuration
            if self.config.network_mode == NetworkMode.NONE:
                container_options['network_mode'] = 'none'
            elif self.config.network_mode == NetworkMode.UNPROXIED:
                container_options['network_mode'] = 'bridge'
            
            # Start container
            print_docker(f"Starting container: {self.container_id}", "info")
            self.container = await asyncio.to_thread(
                self.docker_client.containers.run, **container_options
            )
            
            # Wait for container to be ready
            await self._wait_for_health()
            
            # Verify /results directory exists (should be created by startup command)
            result = await self.exec_command("ls -ld /results")
            if result.exit_code == 0:
                print_docker("Verified /results directory exists in container", "info")
            else:
                # Fallback: create it if somehow it doesn't exist
                await self.exec_command("mkdir -p /results")
                print_docker("Created /results directory in container (fallback)", "warning")
            
            print_docker(f"Container {self.container_id} started successfully", "success")
            return self.container
            
        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to start container: {e}") from e
    
    async def _wait_for_health(self, timeout: int = 30) -> None:
        """Wait for container to be healthy and ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.container:
                    self.container.reload()
                    if self.container.status == 'running':
                        # Try a simple command to verify readiness
                        result = await asyncio.to_thread(
                            self.container.exec_run, 'echo "health check"'
                        )
                        if result.exit_code == 0:
                            print_docker(f"Container {self.container_id} health check passed", "success")
                            return
                        else:
                            print_docker(f"Health check command failed with exit code: {result.exit_code}", "warning")
                    else:
                        print_docker(f"Container status: {self.container.status}", "warning")
                await asyncio.sleep(1)
            except Exception as e:
                print_docker(f"Health check failed: {e}", "warning")
                await asyncio.sleep(1)
        
        # Get container logs for debugging
        if self.container:
            try:
                logs = await asyncio.to_thread(self.container.logs)
                print_docker(f"Container logs: {logs.decode('utf-8', errors='replace')}", "error")
            except Exception as e:
                print_docker(f"Could not get container logs: {e}", "error")
        
        raise RuntimeError(f"Container {self.container_id} failed health check")
    
    async def exec_command(self, command: str, timeout: int = 60) -> ExecutionResult:
        """Execute a command in the container."""
        if not self.container:
            raise RuntimeError("Container not started")
        
        try:
            logger.debug(f"Executing command: {command}")
            result = await asyncio.wait_for(
                asyncio.to_thread(self.container.exec_run, command),
                timeout=timeout
            )
            return ExecutionResult(output=result.output, exit_code=result.exit_code)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Command timed out after {timeout}s: {command}")
    
    async def upload_tar(self, tar_data: bytes, destination: str) -> None:
        """Upload tar archive to container."""
        if not self.container:
            raise RuntimeError("Container not started")
        
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(tar_data)
            tmp_file.flush()
            
            await asyncio.to_thread(
                self.container.put_archive, destination, tmp_file.name
            )
    
    async def download_tar(self, source: str) -> bytes:
        """Download file/directory as tar archive from container."""
        if not self.container:
            raise RuntimeError("Container not started")
        
        tar_stream, _ = await asyncio.to_thread(
            self.container.get_archive, source
        )
        
        # Collect tar data
        tar_data = b''
        for chunk in tar_stream:
            tar_data += chunk
        
        return tar_data
    
    async def export_results(self) -> Optional[str]:
        """Export results from the container to the host using Docker API."""
        if not self.container or not self.config.export_results:
            return None
        
        try:
            import os
            import tarfile
            import tempfile
            
            # Ensure host directory exists
            abs_results_path = os.path.abspath(self.config.results_host_path)
            os.makedirs(abs_results_path, exist_ok=True)
            
            # Check what files exist in /results
            results_ls = await self.exec_command("ls -la /results/")
            print_docker(f"Contents of /results: {results_ls.text_output.strip()}", "info")
            
            # Also check for specific files we expect
            fibonacci_check = await self.exec_command("sh -c 'if [ -f /results/fibonacci.py ]; then ls -la /results/fibonacci.py; else echo fibonacci.py not found; fi'")
            print_docker(f"Fibonacci file check: {fibonacci_check.text_output.strip()}", "info")
            
            # Get tar archive of /results directory
            try:
                tar_data = await self.download_tar("/results")
                print_docker(f"Downloaded tar size: {len(tar_data)} bytes", "info")
                
                if len(tar_data) > 1024:  # More than just directory structure
                    # Extract tar data to host directory
                    with tempfile.NamedTemporaryFile() as tmp_tar:
                        tmp_tar.write(tar_data)
                        tmp_tar.seek(0)
                        
                        with tarfile.open(fileobj=tmp_tar, mode='r') as tar:
                            # List what's in the tar
                            tar_contents = tar.getnames()
                            print_docker(f"Tar contents: {tar_contents}", "info")
                            
                            # Extract all files to the results directory
                            tar.extractall(path=abs_results_path, filter='data')
                    
                    print_docker(f"Results exported to {abs_results_path}", "success")
                    
                    # List what we actually extracted
                    extracted_files = []
                    for root, dirs, files in os.walk(abs_results_path):
                        for file in files:
                            rel_path = os.path.relpath(os.path.join(root, file), abs_results_path)
                            extracted_files.append(rel_path)
                    print_docker(f"Extracted files: {extracted_files}", "info")
                    
                    return abs_results_path
                else:
                    print_docker(f"No files to export (tar size: {len(tar_data)} bytes)", "warning")
                    return None
                
            except Exception as e:
                print_docker(f"Failed to download results: {e}", "warning")
                return None
            
        except Exception as e:
            print_docker(f"Failed to export results: {e}", "warning")
            return None
    
    async def cleanup(self) -> None:
        """Clean up container and allocated resources."""
        try:
            # Export results before cleanup
            if self.container and self.config.export_results:
                await self.export_results()
            
            # Stop and remove container
            if self.container:
                print_docker(f"Cleaning up container: {self.container_id}", "info")
                await asyncio.to_thread(self.container.stop, timeout=10)
                await asyncio.to_thread(self.container.remove, force=True)
                self.container = None
            
            # Release allocated ports
            for port in self.host_ports:
                self.port_manager.release_port(port)
            self.host_ports.clear()
            
        except Exception as e:
            print_docker(f"Error during cleanup: {e}", "error")


# =============================================================================
# DOCKER COMPUTER INTERFACE IMPLEMENTATION
# =============================================================================

class DockerComputerInterface(ComputerInterface):
    """Docker-based implementation of ComputerInterface."""
    
    def __init__(self, container: DockerContainer):
        self.container = container
        self.jupyter_started = False
    
    async def execute_shell(self, command: str, timeout: int = 60) -> ExecutionResult:
        """Execute a shell command in the container."""
        return await self.container.exec_command(command, timeout)
    
    async def execute_python(self, code: str, timeout: int = 60) -> JupyterExecutionResult:
        """Execute Python code via Jupyter kernel."""
        # For simplicity, we'll execute Python code directly via shell
        # In a full implementation, this would use Jupyter kernel protocol
        escaped_code = shlex.quote(code)
        command = f'python3 -c {escaped_code}'
        
        try:
            result = await self.execute_shell(command, timeout)
            status = "success" if result.exit_code == 0 else "failed"
            return JupyterExecutionResult(
                status=status,
                output=result.text_output,
                final_expression_output=None,
                exception={"exit_code": result.exit_code} if result.exit_code != 0 else None
            )
        except Exception as e:
            return JupyterExecutionResult(
                status="failed",
                output=str(e),
                exception={"error": str(e)}
            )
    
    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the container (legacy method)."""
        # This method is kept for compatibility but WriteFileTool now uses Python directly
        content_str = content.decode('utf-8', errors='replace')
        
        # Use Python to write the file instead of shell commands
        python_code = f"""
with open('{destination}', 'w') as f:
    f.write({repr(content_str)})
"""
        result = await self.execute_python(python_code)
        if result.status != "success":
            raise RuntimeError(f"Failed to upload file {destination}: {result.output}")
    
    async def download_file(self, source: str) -> bytes:
        """Download file content from the container."""
        tar_data = await self.container.download_tar(source)
        
        # Extract file content from tar
        with tempfile.NamedTemporaryFile() as tmp_tar:
            tmp_tar.write(tar_data)
            tmp_tar.seek(0)
            
            with tarfile.open(fileobj=tmp_tar, mode='r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            return file_obj.read()
        
        raise RuntimeError(f"Could not extract file: {source}")
    
    async def disable_internet(self) -> None:
        """Disable internet access using iptables."""
        try:
            # Block outbound connections except to local networks
            commands = [
                "iptables -A OUTPUT -d 127.0.0.0/8 -j ACCEPT",
                "iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT", 
                "iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT",
                "iptables -A OUTPUT -d 192.168.0.0/16 -j ACCEPT",
                "iptables -A OUTPUT -j DROP"
            ]
            
            for cmd in commands:
                await self.execute_shell(cmd)
            
            print_docker("Internet access disabled", "success")
        except Exception as e:
            print_docker(f"Failed to disable internet: {e}", "warning")
    
    async def cleanup(self) -> None:
        """Clean up container resources."""
        await self.container.cleanup()


# =============================================================================
# TASK ABSTRACTION AND ORCHESTRATION
# =============================================================================

class Task(BaseModel):
    """Represents a computational task to be executed in a container."""
    task_id: str
    instructions: str
    config: ContainerConfig
    timeout: int = 3600
    task_folder: Optional[str] = None  # Path to task folder for mounting
    _is_temp_folder: bool = False  # Internal flag to track if task_folder is temporary
    
    class Config:
        arbitrary_types_allowed = True
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary task files if they were created."""
        if self._is_temp_folder and self.task_folder and os.path.exists(self.task_folder):
            try:
                shutil.rmtree(self.task_folder, ignore_errors=True)
                print_docker(f"Cleaned up temporary task directory: {self.task_folder}", "info")
            except Exception as e:
                print_docker(f"Failed to clean up temporary directory {self.task_folder}: {e}", "warning")


class TaskLoader:
    """Loads tasks from the tasks/ directory structure."""
    
    def __init__(self, tasks_dir: str = "tasks"):
        self.tasks_dir = Path(tasks_dir)
    
    def list_tasks(self) -> List[str]:
        """List all available task IDs."""
        if not self.tasks_dir.exists():
            return []
        
        tasks = []
        for item in self.tasks_dir.iterdir():
            if item.is_dir() and (item / "instructions.md").exists():
                tasks.append(item.name)
        return tasks
    
    def load_task(self, task_id: str, base_config: ContainerConfig) -> Task:
        """Load a specific task by ID."""
        task_folder = self.tasks_dir / task_id
        instructions_file = task_folder / "instructions.md"
        
        if not task_folder.exists():
            raise ValueError(f"Task folder not found: {task_folder}")
        
        if not instructions_file.exists():
            raise ValueError(f"Instructions file not found: {instructions_file}")
        
        # Read instructions
        instructions = instructions_file.read_text(encoding='utf-8')
        
        # Create a temporary directory for the task files (Docker-accessible)
        # Use a temp directory within the current working directory to ensure Docker can access it
        temp_base_dir = Path.cwd() / "temp_tasks"
        temp_base_dir.mkdir(exist_ok=True)
        temp_task_dir = tempfile.mkdtemp(prefix=f"nanoeval_task_{task_id}_", dir=str(temp_base_dir))
        
        # Copy all task files to the temporary directory
        try:
            for item in task_folder.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_task_dir)
                elif item.is_dir():
                    shutil.copytree(item, Path(temp_task_dir) / item.name)
        except Exception as e:
            # Clean up temp dir if copy fails
            shutil.rmtree(temp_task_dir, ignore_errors=True)
            raise ValueError(f"Failed to copy task files: {e}")
        
        # Create config with temporary task folder mounted
        config = ContainerConfig(
            image=base_config.image,
            environment=base_config.environment.copy() if base_config.environment else {},
            volumes=base_config.volumes.copy() if base_config.volumes else {},
            ports=base_config.ports.copy() if base_config.ports else [],
            privileged=base_config.privileged,
            gpu_access=base_config.gpu_access,
            network_mode=base_config.network_mode,
            memory_limit=base_config.memory_limit,
            timeout=base_config.timeout,
            export_results=base_config.export_results,
            results_host_path=base_config.results_host_path
        )
        
        # Mount the temporary task folder to /task in the container
        config.volumes[temp_task_dir] = "/task"
        
        task = Task(
            task_id=task_id,
            instructions=instructions,
            config=config,
            timeout=base_config.timeout,
            task_folder=temp_task_dir
        )
        task._is_temp_folder = True
        return task
    
    def get_task_instructions(self, task_id: str) -> str:
        """Get just the instructions for a task without loading full config."""
        task_folder = self.tasks_dir / task_id
        instructions_file = task_folder / "instructions.md"
        
        if not instructions_file.exists():
            raise ValueError(f"Instructions file not found: {instructions_file}")
        
        return instructions_file.read_text(encoding='utf-8')


class Step(BaseModel):
    """Represents a step in task execution."""
    step_type: str
    content: str
    timestamp: float = Field(default_factory=time.time)


class FinalResult(BaseModel):
    """Final result of task execution."""
    success: bool
    score: float
    output: str
    execution_time: float
    error: Optional[str] = None


class TaskSolver(ABC):
    """Abstract base class for task solvers."""
    
    @abstractmethod
    async def solve(self, task: Task, computer: ComputerInterface) -> AsyncGenerator[Union[Step, FinalResult], None]:
        """Solve the given task using the computer interface."""
        pass


class SimpleTaskSolver(TaskSolver):
    """Simple example task solver."""
    
    async def solve(self, task: Task, computer: ComputerInterface) -> AsyncGenerator[Union[Step, FinalResult], None]:
        """Example solver that runs some basic commands."""
        start_time = time.time()
        
        try:
            # Step 1: Check environment
            yield Step(step_type="setup", content="Checking environment")
            result = await computer.execute_shell("python3 --version")
            if result.exit_code != 0:
                raise RuntimeError("Python not available")
            
            # Step 2: Create and run a simple program directly
            yield Step(step_type="coding", content="Creating and running test program")
            program = '''
print("Hello from container!")
import sys
print(f"Python version: {sys.version}")
result = 2 + 2
print(f"2 + 2 = {result}")
'''
            
            # Step 3: Execute the program directly (no file upload needed)
            yield Step(step_type="execution", content="Running test program")
            result = await computer.execute_python(program)
            
            # Final result
            execution_time = time.time() - start_time
            yield FinalResult(
                success=result.status == "success",
                score=1.0 if result.status == "success" else 0.0,
                output=result.output,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            yield FinalResult(
                success=False,
                score=0.0,
                output="",
                execution_time=execution_time,
                error=str(e)
            )


# =============================================================================
# REACT AGENT WITH TOOLS (PAPERBENCH STYLE)
# =============================================================================

class ToolCall(BaseModel):
    """Represents a tool call made by the model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    function: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool."""
    tool_call_id: str
    content: str
    success: bool = True
    error: Optional[str] = None


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class ConversationState(BaseModel):
    """Manages the conversation state for the ReAct agent."""
    messages: List[ChatMessage] = Field(default_factory=list)
    max_turns: int = 50
    current_turn: int = 0
    time_limit: Optional[float] = None
    start_time: float = Field(default_factory=time.time)
    
    @property
    def completed(self) -> bool:
        """Check if the conversation should end."""
        if self.current_turn >= self.max_turns:
            return True
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            return True
        return False
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def get_recent_context(self, max_messages: int = 20) -> List[ChatMessage]:
        """Get recent messages for context management."""
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages


class Tool(ABC):
    """Abstract base class for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the model."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass


class BashTool(Tool):
    """Tool for executing bash commands."""
    
    def __init__(self, computer: ComputerInterface):
        self.computer = computer
    
    @property
    def name(self) -> str:
        return "bash"
    
    @property
    def description(self) -> str:
        return """Execute bash commands in the container.
        
        Args:
            cmd (str): The bash command to execute
            
        Returns:
            The output of the command (stdout/stderr)
        """
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute a bash command."""
        cmd = kwargs.get("cmd", "")
        if not cmd:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content="Error: No command provided",
                success=False,
                error="Missing 'cmd' argument"
            )
        
        try:
            result = await self.computer.execute_shell(cmd, timeout=300)  # 5 min timeout
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=f"Exit code: {result.exit_code}\n{result.text_output}",
                success=result.exit_code == 0
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=f"Error executing command: {str(e)}",
                success=False,
                error=str(e)
            )


class PythonTool(Tool):
    """Tool for executing Python code."""
    
    def __init__(self, computer: ComputerInterface):
        self.computer = computer
    
    @property
    def name(self) -> str:
        return "python"
    
    @property
    def description(self) -> str:
        return """Execute Python code in the container.
        
        Args:
            code (str): The Python code to execute
            
        Returns:
            The output of the Python code (stdout/stderr)
        """
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute Python code."""
        code = kwargs.get("code", "")
        if not code:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content="Error: No code provided",
                success=False,
                error="Missing 'code' argument"
            )
        
        try:
            result = await self.computer.execute_python(code, timeout=300)
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=result.output,
                success=result.status == "success"
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=f"Error executing Python code: {str(e)}",
                success=False,
                error=str(e)
            )


class ReadFileTool(Tool):
    """Tool for reading files (paginated like PaperBench)."""
    
    def __init__(self, computer: ComputerInterface):
        self.computer = computer
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return """Read a chunk of lines from a file (paginated).
        
        Args:
            file (str): Path to the file to read
            start_line (int): Line number to start reading from (1-indexed, default: 1)
            max_lines (int): Maximum number of lines to read (default: 50, max: 100)
            
        Returns:
            The requested lines from the file with line numbers
        """
    
    async def execute(self, **kwargs) -> ToolResult:
        """Read a file chunk."""
        file_path = kwargs.get("file", "")
        start_line = kwargs.get("start_line", 1)
        max_lines = min(kwargs.get("max_lines", 50), 100)  # Cap at 100 lines
        
        if not file_path:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content="Error: No file path provided",
                success=False,
                error="Missing 'file' argument"
            )
        
        try:
            # Read the entire file first
            result = await self.computer.execute_shell(f"cat '{file_path}'")
            if result.exit_code != 0:
                return ToolResult(
                    tool_call_id=kwargs.get("tool_call_id", ""),
                    content=f"Error reading file: {result.text_output}",
                    success=False,
                    error=result.text_output
                )
            
            lines = result.text_output.splitlines()
            total_lines = len(lines)
            
            if start_line > total_lines:
                return ToolResult(
                    tool_call_id=kwargs.get("tool_call_id", ""),
                    content=f"Error: start_line ({start_line}) is beyond total lines ({total_lines})",
                    success=False,
                    error="Invalid start_line"
                )
            
            # Get the requested chunk
            end_line = min(start_line + max_lines - 1, total_lines)
            chunk = lines[start_line - 1:end_line]
            
            # Add line numbers
            numbered_lines = [f"{i+start_line}: {line}" for i, line in enumerate(chunk)]
            
            # Add summary
            summary = f"File has {total_lines} total lines. Showing lines {start_line} to {end_line}.\n\n"
            content = summary + "\n".join(numbered_lines)
            
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=content,
                success=True
            )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=f"Error reading file: {str(e)}",
                success=False,
                error=str(e)
            )


class WriteFileTool(Tool):
    """Tool for writing content to files."""
    
    def __init__(self, computer: ComputerInterface):
        self.computer = computer
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return """Write content to a file.
        
        Args:
            file (str): Path to the file to write
            content (str): Content to write to the file
            
        Returns:
            Confirmation of file creation
        """
    
    async def execute(self, **kwargs) -> ToolResult:
        """Write content to a file using Python."""
        file_path = kwargs.get("file", "")
        content = kwargs.get("content", "")
        
        if not file_path:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content="Error: No file path provided",
                success=False,
                error="Missing 'file' argument"
            )
        
        try:
            # Use Python to write the file - this avoids shell escaping issues
            python_code = f"""
import os
# Ensure directory exists
os.makedirs(os.path.dirname('{file_path}'), exist_ok=True)

# Write the file
with open('{file_path}', 'w') as f:
    f.write({repr(content)})

# Verify it was written
import os
size = os.path.getsize('{file_path}')
print(f"Successfully wrote {{size}} bytes to {file_path}")
"""
            
            result = await self.computer.execute_python(python_code)
            
            if result.status == "success":
                # Also verify with ls
                ls_result = await self.computer.execute_shell(f"ls -la '{file_path}'")
                return ToolResult(
                    tool_call_id=kwargs.get("tool_call_id", ""),
                    content=f"{result.output}\n{ls_result.text_output}",
                    success=True
                )
            else:
                return ToolResult(
                    tool_call_id=kwargs.get("tool_call_id", ""),
                    content=f"Python write failed: {result.output}",
                    success=False,
                    error=result.output
                )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=kwargs.get("tool_call_id", ""),
                content=f"Error writing file: {str(e)}",
                success=False,
                error=str(e)
            )


class EndTaskTool(Tool):
    """Tool for ending the task (submission)."""
    
    @property
    def name(self) -> str:
        return "end_task"
    
    @property
    def description(self) -> str:
        return """Signal that you are completely finished with the task.
        
        Args:
            message (str): Optional final message about completion
            
        Returns:
            Confirmation of task completion
        """
    
    async def execute(self, **kwargs) -> ToolResult:
        """End the task."""
        message = kwargs.get("message", "Task completed")
        return ToolResult(
            tool_call_id=kwargs.get("tool_call_id", ""),
            content=f"Task ended: {message}",
            success=True
        )


class MockLLMClient:
    """Mock LLM client that provides hardcoded responses for testing."""
    
    def __init__(self, model: str = "mock"):
        self.model = model
        self.turn_count = 0
    
    async def generate_response(self, messages: List[ChatMessage], tools: List[Tool]) -> ChatMessage:
        """Generate a mock response based on the task."""
        self.turn_count += 1
        
        # Look for task context to determine appropriate response
        last_message = messages[-1] if messages else None
        
        # Check for tool results to determine next step
        has_read_instructions = any(
            msg.role == "tool" and "fibonacci" in msg.content.lower() 
            for msg in messages
        )
        
        has_created_file = any(
            msg.role == "tool" and "fibonacci.py" in msg.content.lower()
            for msg in messages  
        )
        
        if self.turn_count == 1:
            # First turn: read instructions
            return ChatMessage(
                role="assistant",
                content="I'll start by reading the task instructions to understand what needs to be done.",
                tool_calls=[
                    ToolCall(
                        function="read_file",
                        arguments={"file": "/task/instructions.md"}
                    )
                ]
            )
        elif self.turn_count == 2 and has_read_instructions:
            # Second turn: create fibonacci script
            fibonacci_code = '''def fibonacci(n):
    fib_sequence = []
    a, b = 0, 1
    for _ in range(n):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

if __name__ == "__main__":
    fib_numbers = fibonacci(100)
    print("Fibonacci numbers:")
    print(", ".join(map(str, fib_numbers)))'''
            
            return ChatMessage(
                role="assistant",
                content="Now I'll create the fibonacci function and save it to the results directory.",
                tool_calls=[
                    ToolCall(
                        function="write_file",
                        arguments={
                            "file": "/results/fibonacci.py",
                            "content": fibonacci_code
                        }
                    )
                ]
            )
        elif self.turn_count == 3:
            # Third turn: test the code
            return ChatMessage(
                role="assistant", 
                content="Let me test the fibonacci function to make sure it works correctly.",
                tool_calls=[
                    ToolCall(
                        function="python",
                        arguments={
                            "code": '''def fibonacci(n):
    fib_sequence = []
    a, b = 0, 1
    for _ in range(n):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

fib_numbers = fibonacci(100)
print("Fibonacci numbers:")
print(", ".join(map(str, fib_numbers)))'''
                        }
                    )
                ]
            )
        else:
            # Final turn: end task
            return ChatMessage(
                role="assistant",
                content="Perfect! I've successfully created the fibonacci function that generates the first 100 Fibonacci numbers. The script has been saved to /results/fibonacci.py and tested.",
                tool_calls=[
                    ToolCall(
                        function="end_task",
                        arguments={"message": "Fibonacci script created and tested successfully."}
                    )
                ]
            )


class OpenAILLMClient:
    """Real OpenAI client for GPT-4o-mini."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
        except ImportError:
            raise RuntimeError("OpenAI package not installed. Run: pip install openai")
    
    def _format_tools_for_openai(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI API."""
        formatted_tools = []
        for tool in tools:
            # Extract parameter info from description
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add basic parameters based on tool type
            if tool.name == "bash":
                tool_def["function"]["parameters"]["properties"]["cmd"] = {
                    "type": "string",
                    "description": "The bash command to execute"
                }
                tool_def["function"]["parameters"]["required"] = ["cmd"]
            elif tool.name == "python":
                tool_def["function"]["parameters"]["properties"]["code"] = {
                    "type": "string", 
                    "description": "The Python code to execute"
                }
                tool_def["function"]["parameters"]["required"] = ["code"]
            elif tool.name == "read_file":
                tool_def["function"]["parameters"]["properties"] = {
                    "file": {"type": "string", "description": "Path to the file to read"},
                    "start_line": {"type": "integer", "description": "Line to start reading from (1-indexed)", "default": 1},
                    "max_lines": {"type": "integer", "description": "Maximum lines to read", "default": 50}
                }
                tool_def["function"]["parameters"]["required"] = ["file"]
            elif tool.name == "write_file":
                tool_def["function"]["parameters"]["properties"] = {
                    "file": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                }
                tool_def["function"]["parameters"]["required"] = ["file", "content"]
            elif tool.name == "end_task":
                tool_def["function"]["parameters"]["properties"]["message"] = {
                    "type": "string",
                    "description": "Optional completion message"
                }
                tool_def["function"]["parameters"]["required"] = []
            
            formatted_tools.append(tool_def)
        
        return formatted_tools
    
    def _messages_to_openai_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            if msg.role == "system":
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                openai_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                openai_msg = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    openai_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                openai_messages.append(openai_msg)
            elif msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        
        return openai_messages
    
    async def generate_response(self, messages: List[ChatMessage], tools: List[Tool]) -> ChatMessage:
        """Generate response using OpenAI API."""
        try:
            openai_messages = self._messages_to_openai_format(messages)
            openai_tools = self._format_tools_for_openai(tools)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2000
            )
            
            message = response.choices[0].message
            
            # Convert back to our format
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        function=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))
            
            return ChatMessage(
                role="assistant",
                content=message.content or "",
                tool_calls=tool_calls
            )
            
        except Exception as e:
            # Fallback message if API fails
            return ChatMessage(
                role="assistant",
                content=f"I encountered an error calling the API: {str(e)}. Let me try to complete the task anyway.",
                tool_calls=[ToolCall(function="end_task", arguments={"message": f"API Error: {str(e)}"})]
            )


class ReActAgent(TaskSolver):
    """ReAct agent implementation similar to PaperBench."""
    
    def __init__(self, llm_client: Optional[OpenAILLMClient] = None, verbose: bool = True):
        self.llm_client = llm_client or OpenAILLMClient()
        self.tools: Dict[str, Tool] = {}
        self.verbose = verbose
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool
    
    async def solve(self, task: Task, computer: ComputerInterface) -> AsyncGenerator[Union[Step, FinalResult], None]:
        """Solve the task using ReAct pattern with tools."""
        start_time = time.time()
        
        # Initialize tools
        self.add_tool(BashTool(computer))
        self.add_tool(PythonTool(computer))
        self.add_tool(ReadFileTool(computer))
        self.add_tool(WriteFileTool(computer))
        self.add_tool(EndTaskTool())
        
        # Initialize conversation
        state = ConversationState(
            max_turns=20,
            time_limit=task.timeout
        )
        
        # System message
        system_message = ChatMessage(
            role="system",
            content="""You are a helpful AI agent that can use tools to solve programming tasks. 

Available tools:
- bash: Execute bash commands in the container
- python: Execute Python code directly
- read_file: Read files with pagination (supports files in /task and other locations)
- write_file: Write files (save outputs to /results for export)
- end_task: Signal task completion

You should work step by step, using tools to explore, code, test, and verify your solution.
Always test your code to make sure it works correctly before completing the task.

IMPORTANT: 
- You have access to bash, so you can install any dependencies if you need them.
- The task instructions and any supporting files are mounted at /task in the container
- Save any output files to the /results directory so they can be exported to the host system
- You can read /task/instructions.md to see the full task instructions
- The /results directory is automatically mounted and will be available on the host after task completion"""
        )
        state.add_message(system_message)
        
        # Initial user message with task-specific instructions
        user_message = ChatMessage(
            role="user",
            content=f"""Please solve this task: {task.task_id}

The task instructions are available at /task/instructions.md in the container. Please start by reading the instructions to understand what you need to do.

Task preview:
{task.instructions}

Work through this step by step, using the available tools to complete the task."""
        )
        state.add_message(user_message)
        
        if self.verbose:
            print_task_header(task.task_id, list(self.tools.keys()), task.timeout, state.max_turns)
        
        try:
            # Main ReAct loop
            while not state.completed:
                state.current_turn += 1
                
                if self.verbose:
                    print_turn_header(state.current_turn, state.max_turns)
                
                yield Step(
                    step_type="reasoning",
                    content=f"Turn {state.current_turn}: Generating response with {len(self.tools)} tools available"
                )
                
                # Show what we're sending to the LLM
                if self.verbose:
                    print_model_input(state.get_recent_context(), list(self.tools.keys()))
                
                # Get model response
                response = await self.llm_client.generate_response(
                    state.get_recent_context(),
                    list(self.tools.values())
                )
                state.add_message(response)
                
                # Show what we got back from the LLM
                if self.verbose:
                    print_model_output(response.content, response.tool_calls)
                
                # Execute tool calls if any
                if response.tool_calls:
                    yield Step(
                        step_type="tool_execution",
                        content=f"Executing {len(response.tool_calls)} tool call(s): {[tc.function for tc in response.tool_calls]}"
                    )
                    
                    for tool_call in response.tool_calls:
                        if tool_call.function in self.tools:
                            tool = self.tools[tool_call.function]
                            
                            # Execute tool
                            tool_result = await tool.execute(
                                tool_call_id=tool_call.id,
                                **tool_call.arguments
                            )
                            
                            if self.verbose:
                                print_container_action(
                                    f"Tool: {tool_call.function}",
                                    f"Args: {json.dumps(tool_call.arguments, indent=2)}",
                                    tool_result.content,
                                    tool_result.success
                                )
                            
                            # Add tool result to conversation
                            result_message = ChatMessage(
                                role="tool",
                                content=tool_result.content,
                                tool_call_id=tool_call.id
                            )
                            state.add_message(result_message)
                            
                            # Debug: Check files after each tool execution
                            if tool_call.function in ["bash", "python", "write_file"] and self.verbose:
                                try:
                                    debug_ls = await computer.execute_shell("sh -c 'ls -la /results/ 2>/dev/null || echo \"no /results dir\"'")
                                    print_container_action(
                                        "Debug: File System Check",
                                        f"After {tool_call.function} execution",
                                        debug_ls.text_output.strip(),
                                        debug_ls.exit_code == 0
                                    )
                                except:
                                    pass
                            
                            # Check if this was an end_task call
                            if tool_call.function == "end_task":
                                execution_time = time.time() - start_time
                                if self.verbose:
                                    print_final_result(True, 1.0, execution_time, f"Task completed successfully. Tool result: {tool_result.content}")
                                yield FinalResult(
                                    success=True,
                                    score=1.0,
                                    output=f"Task completed successfully. Tool result: {tool_result.content}",
                                    execution_time=execution_time
                                )
                                return
                        else:
                            # Unknown tool
                            error_message = ChatMessage(
                                role="tool",
                                content=f"Error: Unknown tool '{tool_call.function}'",
                                tool_call_id=tool_call.id
                            )
                            state.add_message(error_message)
                            if self.verbose:
                                print_tool(tool_call.function, tool_call.arguments, "error")
                
                # Add progress update
                if state.current_turn % 3 == 0:
                    elapsed = time.time() - start_time
                    print_progress(f"Turn {state.current_turn}, {elapsed:.1f}s elapsed. Continue working or use end_task when complete.")
                
                if self.verbose:
                    print()  # Add spacing between turns
            
            # If we exit the loop without end_task, return final result
            execution_time = time.time() - start_time
            if self.verbose:
                print_final_result(False, 0.5, execution_time, f"Task ended due to limits (turns: {state.current_turn}, time: {execution_time:.1f}s)")
            yield FinalResult(
                success=False,
                score=0.5,  # Partial credit for attempting
                output=f"Task ended due to limits (turns: {state.current_turn}, time: {execution_time:.1f}s)",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            if self.verbose:
                print_final_result(False, 0.0, execution_time, f"Error during execution: {str(e)}")
            yield FinalResult(
                success=False,
                score=0.0,
                output=f"Error during execution: {str(e)}",
                execution_time=execution_time,
                error=str(e)
            )


# =============================================================================
# ORCHESTRATION RUNTIME
# =============================================================================

class ContainerRuntime:
    """Manages the lifecycle of containerized task execution."""
    
    def __init__(self):
        self.port_manager = PortManager()
        self.exit_stack = AsyncExitStack()
    
    @asynccontextmanager
    async def create_computer(self, config: ContainerConfig) -> AsyncGenerator[ComputerInterface, None]:
        """Create and manage a computer interface for the given configuration."""
        container = DockerContainer(config, self.port_manager)
        
        try:
            # Start container
            await container.start()
            
            # Create interface
            computer = DockerComputerInterface(container)
            
            # Register cleanup
            self.exit_stack.push_async_callback(computer.cleanup)
            
            yield computer
            
        except Exception as e:
            await container.cleanup()
            raise e
    
    async def run_task(self, task: Task, solver: TaskSolver) -> List[Union[Step, FinalResult]]:
        """Run a task with the given solver."""
        results = []
        
        try:
            async with self.create_computer(task.config) as computer:
                logger.info(f"Running task {task.task_id}")
                
                async for result in solver.solve(task, computer):
                    results.append(result)
                    logger.info(f"Task {task.task_id}: {result}")
        
        finally:
            # Clean up temporary task files
            task.cleanup_temp_files()
        
        return results
    
    async def cleanup(self) -> None:
        """Clean up all managed resources."""
        await self.exit_stack.aclose()


# =============================================================================
# MULTI-STAGE EVALUATION PIPELINE
# =============================================================================

class EvaluationPipeline:
    """Multi-stage evaluation pipeline similar to PaperBench."""
    
    def __init__(self):
        self.runtime = ContainerRuntime()
    
    async def run_evaluation(self, 
                           agent_config: ContainerConfig,
                           reproduction_config: ContainerConfig,
                           judge_config: ContainerConfig,
                           task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete 3-stage evaluation."""
        
        results = {
            "task_id": task_data.get("id", "unknown"),
            "stages": {},
            "final_score": 0.0
        }
        
        try:
            # Stage 1: Agent Rollout
            logger.info("Stage 1: Agent Rollout")
            agent_task = Task(
                task_id=f"{task_data['id']}-agent",
                config=agent_config
            )
            agent_solver = SimpleTaskSolver()  # Replace with actual agent solver
            agent_results = await self.runtime.run_task(agent_task, agent_solver)
            results["stages"]["agent"] = [r.model_dump() for r in agent_results]
            
            # Stage 2: Reproduction
            logger.info("Stage 2: Reproduction")
            reproduction_task = Task(
                task_id=f"{task_data['id']}-reproduction", 
                config=reproduction_config
            )
            reproduction_solver = SimpleTaskSolver()  # Replace with reproduction solver
            reproduction_results = await self.runtime.run_task(reproduction_task, reproduction_solver)
            results["stages"]["reproduction"] = [r.model_dump() for r in reproduction_results]
            
            # Stage 3: Judging
            logger.info("Stage 3: Judging")
            judge_task = Task(
                task_id=f"{task_data['id']}-judge",
                config=judge_config
            )
            judge_solver = SimpleTaskSolver()  # Replace with judge solver
            judge_results = await self.runtime.run_task(judge_task, judge_solver)
            results["stages"]["judge"] = [r.model_dump() for r in judge_results]
            
            # Calculate final score
            final_results = [r for r in judge_results if isinstance(r, FinalResult)]
            if final_results:
                results["final_score"] = final_results[-1].score
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["error"] = str(e)
        
        finally:
            await self.runtime.cleanup()
        
        return results


# =============================================================================
# UPDATED EXAMPLE USAGE
# =============================================================================

async def run_task_example(task_id: str = None):
    """Example using the PaperBench-style ReAct agent with dynamic task loading."""
    
    # Configure base container with Python and common tools
    base_config = ContainerConfig(
        image="python:3.11-slim",
        environment={
            "PYTHONUNBUFFERED": "1",
            "DEBIAN_FRONTEND": "noninteractive"
        },
        timeout=1800  # 30 minutes
    )
    
    # Load tasks from the tasks/ directory
    task_loader = TaskLoader("tasks")
    available_tasks = task_loader.list_tasks()
    
    if not available_tasks:
        print("❌ No tasks found in tasks/ directory")
        return
    
    print(f"Available tasks: {', '.join(available_tasks)}")
    
    # Use provided task_id or default to first available task
    if task_id not in available_tasks:
        print(f"❌ Task '{task_id}' not found. Available: {', '.join(available_tasks)}")
        return
    else:
        print(f"Running task: {task_id}")
    
    # Load the specific task
    try:
        task = task_loader.load_task(task_id, base_config)
        print(f"✅ Loaded task '{task_id}' from {task.task_folder}")
    except Exception as e:
        print(f"❌ Failed to load task '{task_id}': {e}")
        return
    
    # Create runtime and run ReAct agent
    runtime = ContainerRuntime()
    
    # Use ReAct agent
    try:
        openai_client = OpenAILLMClient(model="gpt-4o-mini")
        react_agent = ReActAgent(llm_client=openai_client, verbose=True)
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        print("Make sure you have OPENAI_API_KEY set in your environment")
        print("Falling back to mock agent...")
        mock_client = MockLLMClient()
        react_agent = ReActAgent(llm_client=mock_client, verbose=True)
    
    try:
        results = await runtime.run_task(task, react_agent)
        
        print(f"=== Task '{task_id}' Results ===")
        for result in results:
            if isinstance(result, Step):
                print(f"[{result.step_type.upper()}] {result.content}")
            elif isinstance(result, FinalResult):
                print_final_result(result.success, result.score, result.execution_time, result.output)
        
    finally:
        await runtime.cleanup()


async def list_tasks_example():
    """Example showing how to list available tasks."""
    task_loader = TaskLoader("tasks")
    available_tasks = task_loader.list_tasks()
    
    if not available_tasks:
        print("❌ No tasks found in tasks/ directory")
        return
    
    print("Available tasks:")
    for task_id in available_tasks:
        try:
            instructions = task_loader.get_task_instructions(task_id)
            # Show first line of instructions as preview
            preview = instructions.split('\n')[0][:80] + ('...' if len(instructions.split('\n')[0]) > 80 else '')
            print(f"  • {task_id}: {preview}")
        except Exception as e:
            print(f"  • {task_id}: (error reading instructions: {e})")



