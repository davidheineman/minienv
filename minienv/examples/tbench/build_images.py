import argparse
import os
import hashlib
import time
import subprocess
import logging
from pathlib import Path
import docker

import subprocess
from typing import Optional
import logging
import subprocess


logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
logger = logging.getLogger(__name__)


BEAKER_USER = "davidh"

# Set Docker platform for compatibility
BEAKER_PLATFORM = "linux/amd64"
os.environ["DOCKER_DEFAULT_PLATFORM"] = BEAKER_PLATFORM


class BeakerImagePusher:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self._logger = logger.getChild(__name__)

    def image_exists_on_beaker(self, image_name: str) -> bool:
        """Check if an image exists on Beaker."""
        beaker_full_name = f"{BEAKER_USER}/{image_name}"
        
        try:
            cmd = ["beaker", "image", "inspect", beaker_full_name]
            self._logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self._logger.info(f"Image {beaker_full_name} exists on Beaker")
            return True
        except subprocess.CalledProcessError:
            self._logger.info(f"Image {beaker_full_name} does not exist on Beaker")
            return False

    def push_local_image(
        self,
        local_image_name: str,
        local_image_tag: str = "latest",
        beaker_image_name: Optional[str] = None,
        beaker_image_tag: str = "latest",
        description: Optional[str] = None,
    ) -> str:
        """Push a local Docker image to Beaker."""
        # Check if local image exists
        local_full_name = f"{local_image_name}:{local_image_tag}"
        if not self._image_exists_locally(local_image_name, local_image_tag):
            raise RuntimeError(f"Local image {local_full_name} not found")

        # Use local image name if beaker name not provided
        if beaker_image_name is None:
            beaker_image_name = local_image_name

        # Generate description if not provided
        if description is None:
            description = f"Pushed from local image {local_full_name}"

        self._logger.info(f"Pushing local image {local_full_name} to Beaker...")

        try:
            cmd = [
                "beaker",
                "image",
                "create",
                "--name",
                beaker_image_name,
                "--workspace",
                self.workspace,
                # "--force",  # Force-push ARM64 images at your peril
                local_full_name,
            ]

            self._logger.info(f"Running: {' '.join(cmd)}")

            # Run the beaker command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            self._logger.info(f"Beaker CLI output: {result.stdout}")

            if result.stderr:
                raise RuntimeError(f"Beaker CLI stderr: {result.stderr}")

            beaker_full_name = (
                f"{self.workspace}/{beaker_image_name}:{beaker_image_tag}"
            )
            self._logger.info(
                f"Successfully pushed image to Beaker: {beaker_full_name}"
            )

            return beaker_full_name

        except subprocess.CalledProcessError as e:
            error_msg = f"Beaker command failed with exit code {e.returncode}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            self._logger.error(error_msg)
            raise RuntimeError(f"Failed to push image to Beaker: {error_msg}")
        except Exception as e:
            self._logger.error(f"Failed to push image to Beaker: {e}")
            raise RuntimeError(f"Failed to push image to Beaker: {e}")

    def _image_exists_locally(self, image_name: str, tag: str) -> bool:
        """Check if a Docker image exists locally."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", f"{image_name}:{tag}"],
                check=True,
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False


class DockerManager:
    def __init__(self):
        """Initialize Docker client."""
        self.client = docker.from_env()
        self.client.ping()
        logger.info("Successfully connected to Docker daemon")

    def build_from_compose(
        self, compose_file_path: str, service_name: str = "client"
    ) -> str:
        compose_path = Path(compose_file_path)
        if not compose_path.exists():
            raise FileNotFoundError(
                f"Docker compose file not found: {compose_file_path}"
            )

        build_context = compose_path.parent
        dockerfile_path = build_context / "Dockerfile"

        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found in: {build_context}")

        logger.info(f"Building Docker image from {dockerfile_path}")

        # Use docker buildx to ensure proper platform targeting
        tag = f"{service_name}-temp"
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            BEAKER_PLATFORM,
            "--tag",
            tag,
            "--load",  # Load the image into docker images
            str(build_context),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info(f"Successfully built image: {tag}")
            return tag
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build Docker image: {e}")

    def tag_image(self, source_tag: str, new_tag: str) -> bool:
        """Tag an existing image with a new tag."""
        cmd = ["docker", "tag", source_tag, new_tag]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully tagged image {source_tag} as {new_tag}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to tag image {source_tag} as {new_tag}: {e}")


def rename_beaker_image(old_name: str, new_name: str):
    logger.info(f"Renaming Beaker image from {old_name} to {new_name}")

    cmd = ["beaker", "image", "rename", old_name, new_name]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    logger.info(f"Successfully renamed Beaker image")


def get_new_task_hash(image_name):
    hash_input = str(time.time()).encode("utf-8")
    sha_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    backup_name = f"{image_name}-{sha_hash}"
    return backup_name


def get_image_name(task_id):
    return f"tb__{task_id}__client"


def build_task(compose_file, task_id, workspace, force_rebuild):
    # Check if image already exists on Beaker
    pusher = BeakerImagePusher(workspace)
    image_name = get_image_name(task_id)
    
    if not force_rebuild and pusher.image_exists_on_beaker(image_name):
        beaker_full_name = f"{BEAKER_USER}/{image_name}"
        logger.info(f"Image for '{task_id}' already exists on Beaker: '{beaker_full_name}'. Skipping...")
        return beaker_full_name
    
    docker_manager = DockerManager()

    # Build and tag image
    temp_tag = docker_manager.build_from_compose(compose_file)
    local_image_tag = get_image_name(task_id)
    docker_manager.tag_image(temp_tag, local_image_tag)

    # Backup existing Beaker image
    try:
        rename_beaker_image(
            old_name=f"{BEAKER_USER}/{get_image_name(task_id)}",
            new_name=get_new_task_hash(get_image_name(task_id)),
        )
    except Exception as e:
        # This will fail if the image doesn't exist yet. Skip in this case.
        pass

    # Push to Beaker
    beaker_image = pusher.push_local_image(
        local_image_name=local_image_tag,
        local_image_tag="latest",
        beaker_image_name=get_image_name(task_id),
        beaker_image_tag="latest",
    )
    logger.info(f"Successfully pushed {task_id} to Beaker: {beaker_image}")

    return beaker_image


def build_tasks(tasks, tasks_dir, workspace, force_rebuild):
    failed_tasks = []
    for task_id in tasks:
        logger.info(f"Building task: {task_id}")

        try:
            compose_file = f"{tasks_dir}/{task_id}/docker-compose.yaml"
            build_task(compose_file, task_id, workspace, force_rebuild)

        except Exception as e:
            logger.error(f"\033[31mFailed to build task {task_id}: {e}\033[0m")
            failed_tasks.append(task_id)

    return failed_tasks


def main(tasks, tasks_dir, workspace, force_rebuild):
    # Ensure all tasks exist and are valid
    if tasks:
        tasks_to_build = [tasks]
    else:
        tasks_path = Path(tasks_dir)
        if not tasks_path.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        tasks_to_build = [
            item.name
            for item in tasks_path.iterdir()
            if item.is_dir() and (item / "docker-compose.yaml").exists()
        ]

        if not tasks_to_build:
            logger.warning(f"No tasks found in {tasks_dir}")
            return

    # Build tasks
    failed_tasks = build_tasks(
        tasks=tasks_to_build, tasks_dir=tasks_dir, workspace=workspace, force_rebuild=force_rebuild
    )

    if failed_tasks:
        raise RuntimeError(f"Failed tasks: {', '.join(failed_tasks)}")

    logger.info("All task images built + pushed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push T-Bench tasks")
    parser.add_argument("--task", type=str, help="Specific task to build")
    parser.add_argument(
        "--tasks-dir", type=str, default="tasks", help="Directory containing tasks"
    )
    parser.add_argument(
        "--workspace", type=str, default="ai2/rollouts", help="Beaker workspace"
    )
    parser.add_argument(
        "-f", "--force-rebuild", action="store_true", help="Force force_rebuild even if image exists on Beaker"
    )

    args = parser.parse_args()

    main(args.task, args.tasks_dir, args.workspace, args.force_rebuild)
