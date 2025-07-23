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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import docker
from docker.models.containers import Container

from . import Backend
from ..constants import TASKS_DIR

logger = logging.getLogger(__name__)


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

    def __init__(self, image: str, environment: Dict[str, str] = None, 
                 volumes: Dict[str, str] = None, ports: List[int] = None,
                 privileged: bool = False, gpu_access: bool = False,
                 memory_limit: Optional[str] = None, timeout: int = 3600,
                 port_manager: PortManager = None):
        self.image = image
        self.environment = environment or {}
        self.volumes = volumes or {}
        self.ports = ports or []
        self.privileged = privileged
        self.gpu_access = gpu_access
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.port_manager = port_manager or PortManager()
        
        self.container: Optional[Container] = None
        self.host_ports: List[int] = []
        self.docker_client = docker.from_env()
        self.container_id = f"minienv-{uuid.uuid4().hex[:8]}"

    async def start(self) -> Container:
        """Start the Docker container with the given configuration."""
        try:
            # Pull image if needed
            logger.info(f"Pulling image: {self.image}")
            await asyncio.to_thread(self.docker_client.images.pull, self.image)

            # Allocate ports
            self.host_ports = [self.port_manager.allocate_port() for _ in self.ports]

            # Configure container options
            container_options = {
                "image": self.image,
                "name": self.container_id,
                "detach": True,
                "environment": self.environment,
                "remove": False,  # We'll remove manually for cleanup control
                "command": [
                    "sh",
                    "-c",
                    "mkdir -p /results && sleep infinity",
                ],  # Create /results and keep container running
            }

            # Port mapping
            if self.ports:
                container_options["ports"] = {
                    f"{container_port}/tcp": host_port
                    for container_port, host_port in zip(self.ports, self.host_ports)
                }

            # Volume mounts
            if self.volumes:
                container_options["volumes"] = {
                    host_path: {"bind": container_path, "mode": "rw"}
                    for host_path, container_path in self.volumes.items()
                }

            # GPU access
            if self.gpu_access:
                container_options["runtime"] = "nvidia"
                container_options["environment"]["NVIDIA_VISIBLE_DEVICES"] = "all"

            # Privileged mode
            if self.privileged:
                container_options["privileged"] = True

            # Memory limit
            if self.memory_limit:
                container_options["mem_limit"] = self.memory_limit

            # Start container
            logger.info(f"Starting container: {self.container_id}")
            self.container = await asyncio.to_thread(
                self.docker_client.containers.run, **container_options
            )

            # Wait for container to be ready
            await self._wait_for_health()

            # Verify /results directory exists
            result = await self.exec_command("ls -ld /results")
            if result[2] == 0:  # exit_code == 0
                logger.info("Verified /results directory exists in container")
            else:
                # Fallback: create it if somehow it doesn't exist
                await self.exec_command("mkdir -p /results")
                logger.warning("Created /results directory in container (fallback)")

            logger.info(f"Container {self.container_id} started successfully")
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
                    if self.container.status == "running":
                        # Try a simple command to verify readiness
                        result = await asyncio.to_thread(
                            self.container.exec_run, 'echo "health check"'
                        )
                        if result.exit_code == 0:
                            logger.info(f"Container {self.container_id} health check passed")
                            return
                        else:
                            logger.warning(f"Health check command failed with exit code: {result.exit_code}")
                    else:
                        logger.warning(f"Container status: {self.container.status}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                await asyncio.sleep(1)

        # Get container logs for debugging
        if self.container:
            try:
                logs = await asyncio.to_thread(self.container.logs)
                logger.error(f"Container logs: {logs.decode('utf-8', errors='replace')}")
            except Exception as e:
                logger.error(f"Could not get container logs: {e}")

        raise RuntimeError(f"Container {self.container_id} failed health check")

    async def exec_command(self, command: str, timeout: int = 60) -> Tuple[str, str, int]:
        """Execute a command in the container and return (stdout, stderr, exit_code)."""
        if not self.container:
            raise RuntimeError("Container not started")

        try:
            logger.debug(f"Executing command: {command}")
            result = await asyncio.wait_for(
                asyncio.to_thread(self.container.exec_run, command), timeout=timeout
            )
            output_text = result.output.decode("utf-8", errors="replace")
            # Docker exec_run combines stdout and stderr, so we return it as stdout
            return output_text, "", result.exit_code
        except asyncio.TimeoutError:
            raise RuntimeError(f"Command timed out after {timeout}s: {command}")

    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the container."""
        if not self.container:
            raise RuntimeError("Container not started")

        # Create a tar archive with the file
        with tempfile.NamedTemporaryFile() as tmp_tar:
            with tarfile.open(fileobj=tmp_tar, mode='w') as tar:
                # Create file info
                tarinfo = tarfile.TarInfo(name=os.path.basename(destination))
                tarinfo.size = len(content)
                tarinfo.mode = 0o644
                
                # Create a temporary file with the content
                with tempfile.NamedTemporaryFile(mode='w+b') as content_file:
                    content_file.write(content)
                    content_file.seek(0)
                    
                    # Add file to tar
                    tar.addfile(tarinfo, fileobj=content_file)
                
            tmp_tar.seek(0)
            
            # Upload to container
            destination_dir = os.path.dirname(destination)
            await asyncio.to_thread(self.container.put_archive, destination_dir, tmp_tar.read())

    async def download_file(self, source: str) -> bytes:
        """Download file content from the container."""
        if not self.container:
            raise RuntimeError("Container not started")

        tar_stream, _ = await asyncio.to_thread(self.container.get_archive, source)

        # Collect tar data
        tar_data = b""
        for chunk in tar_stream:
            tar_data += chunk

        # Extract file content from tar
        with tempfile.NamedTemporaryFile() as tmp_tar:
            tmp_tar.write(tar_data)
            tmp_tar.seek(0)

            with tarfile.open(fileobj=tmp_tar, mode="r") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            return file_obj.read()

        raise RuntimeError(f"Could not extract file: {source}")

    async def export_results(self, results_host_path: str = "./minienv_results") -> Optional[str]:
        """Export results from the container to the host using Docker API."""
        if not self.container:
            return None

        try:
            # Ensure host directory exists
            abs_results_path = os.path.abspath(results_host_path)
            os.makedirs(abs_results_path, exist_ok=True)

            # Check what files exist in /results
            results_ls, _, _ = await self.exec_command("ls -la /results/")
            logger.info(f"Contents of /results: {results_ls.strip()}")

            # Get tar archive of /results directory
            try:
                tar_data = await self.download_tar("/results")
                logger.info(f"Downloaded tar size: {len(tar_data)} bytes")

                if len(tar_data) > 1024:  # More than just directory structure
                    # Extract tar data to host directory
                    with tempfile.NamedTemporaryFile() as tmp_tar:
                        tmp_tar.write(tar_data)
                        tmp_tar.seek(0)

                        with tarfile.open(fileobj=tmp_tar, mode="r") as tar:
                            # List what's in the tar
                            tar_contents = tar.getnames()
                            logger.info(f"Tar contents: {tar_contents}")

                            # Extract all files to the results directory
                            tar.extractall(path=abs_results_path, filter="data")

                    logger.info(f"Results exported to {abs_results_path}")

                    # List what we actually extracted
                    extracted_files = []
                    for root, dirs, files in os.walk(abs_results_path):
                        for file in files:
                            rel_path = os.path.relpath(os.path.join(root, file), abs_results_path)
                            extracted_files.append(rel_path)
                    logger.info(f"Extracted files: {extracted_files}")

                    return abs_results_path
                else:
                    logger.warning(f"No files to export (tar size: {len(tar_data)} bytes)")
                    return None

            except Exception as e:
                logger.warning(f"Failed to download results: {e}")
                return None

        except Exception as e:
            logger.warning(f"Failed to export results: {e}")
            return None

    async def download_tar(self, source: str) -> bytes:
        """Download file/directory as tar archive from container."""
        if not self.container:
            raise RuntimeError("Container not started")

        tar_stream, _ = await asyncio.to_thread(self.container.get_archive, source)

        # Collect tar data
        tar_data = b""
        for chunk in tar_stream:
            tar_data += chunk

        return tar_data

    async def cleanup(self) -> None:
        """Clean up container and allocated resources."""
        try:
            # Stop and remove container
            if self.container:
                logger.info(f"Cleaning up container: {self.container_id}")
                await asyncio.to_thread(self.container.stop, timeout=10)
                await asyncio.to_thread(self.container.remove, force=True)
                self.container = None

            # Release allocated ports
            for port in self.host_ports:
                self.port_manager.release_port(port)
            self.host_ports.clear()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class LocalBackend(Backend):
    """Local Docker backend implementation."""

    def __init__(self, 
                 environment: Dict[str, str] = None,
                 volumes: Dict[str, str] = None,
                 ports: List[int] = None,
                 privileged: bool = False,
                 gpu_access: bool = False,
                 memory_limit: Optional[str] = None,
                 timeout: int = 3600,
                 results_host_path: str = "./minienv_results"):
        self.environment = environment or {"PYTHONUNBUFFERED": "1"}
        self.volumes = volumes or {}
        self.ports = ports or []
        self.privileged = privileged
        self.gpu_access = gpu_access
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.results_host_path = results_host_path
        
        self.container: Optional[DockerContainer] = None
        self.port_manager = PortManager()
        self.temp_task_dir: Optional[str] = None

    async def create_env(self, task_name: str, image: str, **kwargs) -> None:
        """Create and initialize the Docker environment."""
        existing_task_folder = kwargs.get('existing_task_folder')
        
        if existing_task_folder and os.path.exists(existing_task_folder):
            # Use the existing task folder provided by the runner
            self.temp_task_dir = existing_task_folder
            self._using_external_task_folder = True
        else:
            # Create a temporary directory for the task files (fallback)
            temp_base_dir = Path.cwd() / "temp_tasks"
            temp_base_dir.mkdir(exist_ok=True)
            self.temp_task_dir = tempfile.mkdtemp(prefix=f"minienv_task_{task_name}_", dir=str(temp_base_dir))

            # Copy task files to the temporary directory
            task_folder = Path(TASKS_DIR) / task_name
            if task_folder.exists():
                try:
                    for item in task_folder.iterdir():
                        if item.is_file():
                            shutil.copy2(item, self.temp_task_dir)
                        elif item.is_dir():
                            shutil.copytree(item, Path(self.temp_task_dir) / item.name)
                except Exception as e:
                    # Clean up temp dir if copy fails
                    shutil.rmtree(self.temp_task_dir, ignore_errors=True)
                    raise ValueError(f"Failed to copy task files: {e}")

        # Set up volumes to mount the task directory
        volumes = self.volumes.copy()
        volumes[self.temp_task_dir] = "/task"

        # Create and start container
        self.container = DockerContainer(
            image=image,
            environment=self.environment,
            volumes=volumes,
            ports=self.ports,
            privileged=self.privileged,
            gpu_access=self.gpu_access,
            memory_limit=self.memory_limit,
            timeout=self.timeout,
            port_manager=self.port_manager
        )

        await self.container.start()

    async def exec_command(self, command: str, timeout: int = 60) -> Tuple[str, str, int]:
        """Execute a command and return (stdout, stderr, exit_code)."""
        if not self.container:
            raise RuntimeError("Environment not created")
        
        return await self.container.exec_command(command, timeout)

    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the environment."""
        if not self.container:
            raise RuntimeError("Environment not created")
        
        await self.container.upload_file(content, destination)

    async def download_file(self, source: str) -> bytes:
        """Download file content from the environment."""
        if not self.container:
            raise RuntimeError("Environment not created")
        
        return await self.container.download_file(source)

    async def teardown(self) -> bool:
        """Clean up the environment."""
        success = True
        
        try:
            # Export results before cleanup
            if self.container:
                await self.container.export_results(self.results_host_path)
                await self.container.cleanup()
                self.container = None
        except Exception as e:
            logger.error(f"Error during container cleanup: {e}")
            success = False

        # Clean up temporary task files only if we created them
        if (self.temp_task_dir and os.path.exists(self.temp_task_dir) and 
            not hasattr(self, '_using_external_task_folder')):
            try:
                shutil.rmtree(self.temp_task_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary task directory: {self.temp_task_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {self.temp_task_dir}: {e}")
                success = False

        return success 