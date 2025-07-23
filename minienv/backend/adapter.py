import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from minienv.runner import ComputerInterface, ExecutionResult, JupyterExecutionResult
from . import Backend


class BackendComputerInterface(ComputerInterface):
    """Adapter that bridges Backend to ComputerInterface."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self._jupyter_session_id = 0

    async def execute_shell(self, command: str, timeout: int = 60) -> ExecutionResult:
        """Execute a shell command in the backend environment."""
        # Handle different backend command formats
        if hasattr(self.backend, '__class__') and 'Beaker' in self.backend.__class__.__name__:
            # Beaker backend expects a list of command parts
            import shlex
            command_list = shlex.split(command)
            stdout, stderr, exit_code = await self.backend.exec_command(command_list, timeout)
        else:
            # Local backend expects a string
            stdout, stderr, exit_code = await self.backend.exec_command(command, timeout)
        
        # Combine stdout and stderr for output
        combined_output = stdout
        if stderr.strip():
            combined_output += f"\n{stderr}"
        
        return ExecutionResult(
            output=combined_output.encode('utf-8'),
            exit_code=exit_code
        )

    async def execute_python(self, code: str, timeout: int = 60) -> JupyterExecutionResult:
        """Execute Python code in the backend environment."""
        # Create a temporary Python script
        self._jupyter_session_id += 1
        script_name = f"/tmp/jupyter_exec_{self._jupyter_session_id}.py"
        
        try:
            # Create the directory first
            await self.backend.exec_command("mkdir -p /tmp", 5)
            
            # Upload the Python code as a script
            await self.backend.upload_file(code.encode('utf-8'), script_name)
            
            # Execute the script with Python
            if hasattr(self.backend, '__class__') and 'Beaker' in self.backend.__class__.__name__:
                import shlex
                python_cmd = shlex.split(f"python3 {script_name}")
                stdout, stderr, exit_code = await self.backend.exec_command(python_cmd, timeout)
                
                # Clean up the temporary script
                try:
                    cleanup_cmd = shlex.split(f"rm -f {script_name}")
                    await self.backend.exec_command(cleanup_cmd, timeout=5)
                except:
                    pass  # Ignore cleanup errors
            else:
                stdout, stderr, exit_code = await self.backend.exec_command(
                    f"python3 {script_name}", timeout
                )
                
                # Clean up the temporary script
                try:
                    await self.backend.exec_command(f"rm -f {script_name}", timeout=5)
                except:
                    pass  # Ignore cleanup errors
            
            # Determine status based on exit code
            status = "success" if exit_code == 0 else "error"
            
            # Combine stdout and stderr for output
            output = stdout
            if stderr.strip():
                output += f"\n{stderr}"
            
            # For simplicity, we don't separate final expression output
            return JupyterExecutionResult(
                status=status,
                output=output,
                final_expression_output=None,
                exception={"message": stderr} if exit_code != 0 and stderr else None
            )
            
        except Exception as e:
            return JupyterExecutionResult(
                status="error",
                output=f"Error executing Python code: {str(e)}",
                final_expression_output=None,
                exception={"message": str(e)}
            )

    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the backend environment."""
        await self.backend.upload_file(content, destination)

    async def download_file(self, source: str) -> bytes:
        """Download file content from the backend environment."""
        return await self.backend.download_file(source)

    async def disable_internet(self) -> None:
        """Disable internet access for the environment."""
        # This is a no-op for now - could be implemented per backend
        pass

    async def cleanup(self) -> None:
        """Clean up backend resources."""
        await self.backend.teardown() 