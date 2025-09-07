import asyncio
import io
import logging
import os
import shlex
import subprocess
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

os.environ["GRPC_VERBOSITY"] = "ERROR" # silence beaker GPRC warnings

# harden the GRPC connection
os.environ["GRPC_KEEPALIVE_TIME_MS"] = "30000"
os.environ["GRPC_KEEPALIVE_TIMEOUT_MS"] = "20000"
os.environ["GRPC_GO_IDLE_CONNECTION_TIMEOUT"] = "0"

# Don't propagate tenacity failures
logging.getLogger("tenacity").propagate = False

import docker
from docker.models.containers import Container

from minienv.backend.beaker import (
    BeakerBackend,
    get_hostname,
    launch_beaker_job,
)

from rich.console import Console

from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.utils.logger import logger

console = Console()

BEAKER_USER = "davidh"


class DirectDockerBackend(BeakerBackend):
    """Direct Docker backend that connects to remote Docker daemon via SSH."""

    def __init__(
        self,
        workspace: str = "ai2/rollouts",
        hostname: Optional[str] = None,
        username: str = BEAKER_USER,
    ):
        super().__init__(workspace)
        self.job = None
        self.docker_client = None
        self.container = None
        self.hostname = hostname
        self.username = username

    def _get_container(self) -> Container:
        """Get the running container from the remote Docker daemon."""
        if not self.docker_client:
            raise RuntimeError("Docker client not initialized")

        # Get containers and find the one with the specific beaker task label
        containers = self.docker_client.containers.list()
        if not containers:
            raise RuntimeError("No running containers found on remote host")

        # Get the container corresponding to the task ID
        task_id = self.job.task_id
        
        for container in containers:
            labels = container.attrs.get("Config", {}).get("Labels", {})
            if labels.get("beaker.org/task") == task_id:
                return container
        
        raise RuntimeError(f"No container found with beaker.org/task label '{task_id}'")

    def _add_host_to_known_hosts(self, hostname: str):
        """Add host to SSH known_hosts if not already present."""    
        # Get ~/.ssh/known_hosts
        known_hosts_path = Path.home() / ".ssh" / "known_hosts"
        known_hosts_path.parent.mkdir(mode=0o700, exist_ok=True)
        
        # Check if host is already in known_hosts
        if known_hosts_path.exists():
            try:
                result = subprocess.run(
                    ["ssh-keygen", "-F", hostname],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.debug(f"Host {hostname} already in known_hosts")
                    return
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass  # Continue to add the host
        
        # Add host to known_hosts using ssh-keyscan
        try:
            logger.debug(f"Adding {hostname} to known_hosts")
            result = subprocess.run(
                ["ssh-keyscan", "-H", hostname],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Append to known_hosts file
                with open(known_hosts_path, "a") as f:
                    f.write(result.stdout)
                logger.debug(f"Successfully added {hostname} to known_hosts")
            else:
                logger.warning(f"Failed to get host key for {hostname}: {result.stderr}")
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Failed to add {hostname} to known_hosts: {e}")

    def _initialize_docker_connection(self, hostname: str, username: str = BEAKER_USER):
        """Initialize the Docker client connection to the remote daemon via SSH."""
        self.hostname = hostname
        self.username = username
        
        # Ensure the host is in known_hosts to avoid SSH verification errors
        self._add_host_to_known_hosts(hostname)
        
        # Use Docker over SSH directly (no TCP tunnel needed)
        ssh_host = f"ssh://{username}@{hostname}"
        logger.debug(f"Connecting to Docker via SSH: {ssh_host}")
        
        # Create docker client with backoffs
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
        def _create_docker_client():
            return docker.DockerClient(
                base_url=ssh_host,
                timeout=300,
            )
        
        self.docker_client = _create_docker_client()

        # Test the connection
        try:
            info = self.docker_client.info()
            logger.debug(
                f"Successfully connected to remote Docker daemon via SSH: {info.get('Name', 'Unknown')} ({username}@{hostname})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to remote Docker daemon via SSH: {str(e)}"
            )

        # Get the container
        self.container = self._get_container()

    async def exec_command(
        self, command: str | list[str], timeout: int = 300, cwd: str = None
    ) -> tuple[str, str, int]:
        """Execute command directly via Docker client to the remote container."""
        if self.job is None:
            raise RuntimeError("Backend not initialized: no job available")

        # Convert command to list if it's a string
        if isinstance(command, str):
            command = shlex.split(command)

        print(shlex.join(command))

        assert self.container is not None, "Need an initialized container!"

        try:
            # Execute command in the container
            exec_result = self.container.exec_run(
                cmd=command,
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                privileged=False,
                user="",
                workdir=cwd,
            )

            # Decode the output
            output = exec_result.output.decode("utf-8") if exec_result.output else ""

            # Docker exec_run combines stdout and stderr, so we'll split them
            # For now, we'll put everything in stdout and leave stderr empty
            # You could enhance this by parsing the output or using separate streams
            stdout = output
            stderr = ""
            returncode = exec_result.exit_code

            return stdout, stderr, returncode

        except docker.errors.APIError as e:
            return "", f"Docker API error: {str(e)}", 1
        except Exception as e:
            return "", f"Command execution failed: {str(e)}", 1

    async def upload_file(self, content: bytes, destination: str) -> None:
        """Upload file content to the remote container via Docker client."""
        if self.job is None:
            raise RuntimeError("Backend not initialized: no job available")

        try:
            # Create a tar archive with the file content
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=os.path.basename(destination))
                tarinfo.size = len(content)
                tar.addfile(tarinfo, io.BytesIO(content))

            tar_stream.seek(0)

            # Upload to container
            parent_dir = os.path.dirname(destination)
            if not parent_dir:
                parent_dir = "/"

            self.container.put_archive(path=parent_dir, data=tar_stream.getvalue())

        except docker.errors.APIError as e:
            raise RuntimeError(f"Docker API error during file upload: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"File upload failed: {str(e)}")


class MinienvContainer:
    def __init__(self, backend: DirectDockerBackend):
        self._backend = backend

    def exec_run(self, cmd: list[str], user=""):
        class ExecResult:
            def __init__(self, output: str, exit_code: int):
                self.output = output.encode()
                self.exit_code = exit_code

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            stdout, stderr, returncode = loop.run_until_complete(
                self._backend.exec_command(cmd, timeout=30)
            )
            # Combine stdout and stderr like Docker does
            combined_output = stdout
            if stderr:
                combined_output += "\n" + stderr
            return ExecResult(combined_output, returncode)
        finally:
            loop.close()

    def put_archive(self, path: str, data: bytes):
        # Ensure we have a container reference
        if not self._backend.container:
            self._backend.container = self._backend._get_container()

        self._backend.container.put_archive(path=path, data=data)


class MinienvSession(TmuxSession):
    """A Session implementation that uses tmux inside minienv containers."""

    def __init__(
        self,
        session_name: str,
        backend,
        commands_path: Path | None = None,
        disable_recording: bool = False,
        user: str = "",
    ):
        self._backend = backend
        self._session_id = None

        # Create a mock container that adapts our SSH backend to TmuxSession interface
        mock_container = MinienvContainer(backend)

        # Call parent constructor with the mock container
        super().__init__(
            session_name=session_name,
            container=mock_container,
            commands_path=commands_path,
            disable_recording=disable_recording,
            user=user,
        )

    @property
    def logging_path(self) -> Path:
        return (
            Path(MinienvTerminal.CONTAINER_SESSION_LOGS_PATH)
            / f"{self._session_name}.log"
        )

    def start(self) -> None:
        # Ensure logging dir exists (minienv-specific)
        asyncio.run(self._backend.exec_command("mkdir -p /logs"))

        super().start()
        self._session_id = f"{self._session_name}_{int(time.time())}"

    def stop(self) -> None:
        super().stop()
        self._session_id = None

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ):
        super().send_keys(
            keys=keys,
            block=block,
            min_timeout_sec=min_timeout_sec,
            max_timeout_sec=max_timeout_sec,
        )
        self._sync_session_logs_if_possible()

    def send_command(self, command: TerminalCommand) -> None:
        super().send_command(command)
        self._sync_session_logs_if_possible()

    def sync_logs(self, sessions_logs_path: Path | None = None) -> None:
        """Sync this session's logs to local filesystem."""
        if not sessions_logs_path:
            return

        try:
            log_file_path = str(self.logging_path)
            stdout, stderr, returncode = asyncio.run(
                self._backend.exec_command(f"cat {log_file_path}")
            )
            log_content = stdout

            local_log_path = sessions_logs_path / f"{self._session_name}.log"
            local_log_path.parent.mkdir(parents=True, exist_ok=True)
            local_log_path.write_text(log_content)
            self._logger.debug(
                f"Synced session log: {log_file_path} -> {local_log_path}"
            )

        except Exception as e:
            self._logger.error(
                f"Failed to sync logs for session {self._session_name}: {e}"
            )

    def _sync_session_logs_if_possible(self) -> None:
        """Try to sync session logs, but don't fail if we can't."""
        try:
            self.sync_logs(self._terminal_sessions_logs_path)
        except Exception as e:
            self._logger.debug(f"Could not sync logs immediately: {e}")


class MinienvTerminal:
    """A terminal using a Beaker backend"""

    # Container paths that match DockerComposeManager constants
    CONTAINER_SESSION_LOGS_PATH = "/logs"
    CONTAINER_AGENT_LOGS_PATH = "/agent-logs"
    CONTAINER_TEST_DIR = Path("/tests")

    def __init__(
        self,
        client_container_name: str,
        client_image_name: str,
        docker_compose_path: Path,
        docker_image_name_prefix: str | None = None,
        sessions_logs_path: Path | None = None,
        agent_logs_path: Path | None = None,
        commands_path: Path | None = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        livestream: bool = False,
        disable_recording: bool = False,
        beaker_workspace: str | None = "ai2/rollouts",
    ):
        self._client_container_name = client_container_name
        self._client_image_name = client_image_name
        self._docker_compose_path = docker_compose_path
        self._sessions_logs_path = sessions_logs_path
        self._agent_logs_path = agent_logs_path
        self._commands_path = commands_path
        self._livestream = livestream
        self._disable_recording = disable_recording
        self._cleanup = cleanup
        self._beaker_workspace = beaker_workspace
        self._logger = logger.getChild(__name__)
        self._sessions: dict[str, MinienvSession] = {}

        # Use DirectDockerBackend only
        self._backend = DirectDockerBackend(workspace=beaker_workspace)

        self._environment_id = None

        # Initialize log directories if paths are provided
        if self._sessions_logs_path:
            self._sessions_logs_path.mkdir(parents=True, exist_ok=True)
        if self._agent_logs_path:
            self._agent_logs_path.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the minienv environment."""
        try:
            # TODO @davidh: Support all this functionality natively in minienv
            task_name = self._client_container_name
            image = self._client_image_name

            # TODO: read self._docker_compose_path (yaml) for the env vars in services.client.environment

            image = "davidh/" + image

            # TODO: Ensure that the image exists in the workspace

            job = launch_beaker_job(
                name=f"tbench.{task_name}",
                description=f"Terminal-Bench: '{task_name}' on '{image}'",
                docker_image=image,
                # additional_mounts={
                #     # - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
                #     # - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
                #     "/var/log/tbench": T_BENCH_CONTAINER_LOGS_PATH,
                # },  # TODO: add volumes here
                # result_path="/var/log/tbench",
                entrypoint=[
                    "sh",
                    "-c",
                    "trap 'exit 0' TERM INT; while true; do sleep 1; done",
                ],
                # TODO: Use these to build and deploy the image
                # self.client_container_name
                # self.client_image_name
                env_vars={  # TODO: Load env vars from container (maybe you can inspect the image to get this?)
                    "TEST_DIR": "/tests",
                },
                workspace=self._beaker_workspace,
            )

            hostname = get_hostname(job)

            self._backend.job = job
            self._backend.image_name = image

            self.hostname = hostname

            # Initialize Docker connection via SSH using the hostname from Beaker
            self._backend._initialize_docker_connection(
                hostname=hostname, username=BEAKER_USER
            )
            self._logger.debug(
                f"Initialized Docker connection via SSH to {BEAKER_USER}@{hostname}"
            )

            self._environment_id = "started"

        except Exception as e:
            self._logger.error(f"Failed to start minienv environment: {e}")
            raise

    def stop(self) -> None:
        for session in self._sessions.values():
            session.stop()

        self.sync_all_logs()

        if self._environment_id:
            try:
                # Shut down the parent process for the container
                session = self.create_session("shutdown")
                session.send_command(TerminalCommand(command="kill -TERM 1"))
            except Exception as e:
                self._logger.error(f"Error stopping minienv environment: {e}")

        # Reset sessions dict
        self._sessions.clear()
        self._environment_id = None

    def create_session(
        self,
        session_name: str,
        is_active_stream: bool = False,
        as_configured_user: bool = True,
    ) -> MinienvSession:
        if self._environment_id is None:
            raise ValueError("Environment not started. Run start() first.")

        if session_name in self._sessions:
            raise ValueError(f"Session {session_name} already exists")

        # Determine user
        user = "root" if not as_configured_user else ""

        session = MinienvSession(
            session_name=session_name,
            backend=self._backend,
            commands_path=self._commands_path,
            disable_recording=self._disable_recording,
            user=user,
        )

        # Pass the sessions_logs_path to the session for immediate syncing
        session._terminal_sessions_logs_path = self._sessions_logs_path

        self._sessions[session_name] = session

        if is_active_stream:
            self.set_active_stream(session_name)

        session.start()
        self.sync_session_logs()

        return session

    def get_session(self, session_name: str) -> MinienvSession:
        if session_name not in self._sessions:
            raise ValueError(f"Session {session_name} does not exist")
        return self._sessions[session_name]

    def copy_to_container(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ):
        """Copy files to the minienv environment."""
        if isinstance(paths, Path):
            paths = [paths]

        async def copy_file(src_path: Path, dest_path: str):
            parent_dir = str(Path(dest_path).parent)
            await self._backend.exec_command(f"mkdir -p {parent_dir}")
            content = src_path.read_bytes()
            await self._backend.upload_file(content, dest_path)

        async def copy_all():
            for path in paths:
                if path.is_file():
                    dest_path = (
                        f"{container_dir or '/tmp'}/{container_filename or path.name}"
                    )
                    await copy_file(path, dest_path)
                    self._logger.debug(
                        f"Successfully copied file {path} to {dest_path}"
                    )

                elif path.is_dir():
                    dest_path = container_dir or "/tmp"
                    await self._backend.exec_command(f"mkdir -p {dest_path}")

                    for item in path.rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(path)
                            item_dest = f"{dest_path}/{relative_path}"
                            await copy_file(item, item_dest)
                            self._logger.debug(f"Copied {item} to {item_dest}")

                    self._logger.debug(
                        f"Successfully copied directory contents from {path} to {dest_path}"
                    )

        try:
            asyncio.run(copy_all())
        except Exception as e:
            self._logger.error(f"Failed to copy {paths}: {e}")

    def _sync_logs_from_container(self, container_path: str, local_path: Path) -> None:
        """Helper method to sync logs from a container directory to local filesystem."""
        try:
            stdout, stderr, returncode = asyncio.run(
                self._backend.exec_command(
                    f"find {container_path} -type f 2>/dev/null || true"
                )
            )

            if returncode == 0 and stdout.strip():
                for log_file in stdout.strip().split("\n"):
                    if not log_file.strip():
                        continue

                    rel_path = Path(log_file).relative_to(container_path)
                    local_file_path = local_path / rel_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    content, _, ret = asyncio.run(
                        self._backend.exec_command(f"cat {log_file}")
                    )
                    if ret == 0:
                        local_file_path.write_text(content)

        except Exception as e:
            self._logger.error(f"Failed to sync logs from {container_path}: {e}")

    def sync_session_logs(self) -> None:
        """Sync session logs from container to local filesystem."""
        if not self._sessions_logs_path:
            return

        # Sync log files from container
        self._sync_logs_from_container(
            self.CONTAINER_SESSION_LOGS_PATH, self._sessions_logs_path
        )

        # Also sync active tmux sessions
        try:
            for session_name in self._sessions:
                capture, _, ret = asyncio.run(
                    self._backend.exec_command(
                        f"tmux capture-pane -t {session_name} -p 2>/dev/null || true"
                    )
                )
                if ret == 0 and capture.strip():
                    local_path = self._sessions_logs_path / f"{session_name}.log"
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    if not local_path.exists() or len(capture) > len(
                        local_path.read_text()
                    ):
                        local_path.write_text(capture)
        except Exception as e:
            self._logger.error(f"Failed to sync active tmux sessions: {e}")

    def sync_agent_logs(self) -> None:
        """Sync agent logs from container to local filesystem."""
        self._sync_logs_from_container(
            self.CONTAINER_AGENT_LOGS_PATH, self._agent_logs_path
        )

    def sync_all_logs(self) -> None:
        """Sync both session and agent logs from container to local filesystem."""
        self.sync_session_logs()
        self.sync_agent_logs()

    def get_local_session_log_path(self, session_name: str) -> Path | None:
        """Get the local path where a session's logs are stored."""
        if not self._sessions_logs_path:
            return None
        return self._sessions_logs_path / f"{session_name}.log"

    def get_local_agent_log_path(self, relative_path: str) -> Path | None:
        """Get the local path where agent logs are stored."""
        if not self._agent_logs_path:
            return None
        return self._agent_logs_path / relative_path

    def set_active_stream(self, session_name: str) -> None:
        raise NotImplementedError()


@contextmanager
def spin_up_minienv_terminal(
    client_container_name: str,
    client_image_name: str,
    docker_compose_path: Path,
    **kwargs,
) -> Generator[MinienvTerminal, None, None]:

    terminal = MinienvTerminal(
        client_container_name=client_container_name,
        client_image_name=client_image_name,
        docker_compose_path=docker_compose_path,
        **kwargs,
    )

    try:
        terminal.start()
        yield terminal
    finally:
        terminal.stop()
