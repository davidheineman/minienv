import argparse
import os
import subprocess
import multiprocessing
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
logger = logging.getLogger(__name__)

BEAKER_USER = "davidh"
BEAKER_PLATFORM = "linux/amd64"


def get_image_name(task_id):
    return f"tb__{task_id}__client"


def verify_container_files(beaker_image: str, task_id: str, tasks_dir: str) -> bool:
    """Verify that the container has the correct files by comparing paper.md."""
    try:
        # Pull the fresh image from Beaker
        logger.info(f"Pulling fresh image from Beaker: {beaker_image}")
        pull_cmd = ["beaker", "image", "pull", beaker_image]
        result = subprocess.run(pull_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully pulled image: {beaker_image}")
        
        # Get the local paper.md file path
        local_paper_path = Path(tasks_dir) / task_id / "paper.md"
        if not local_paper_path.exists():
            logger.warning(f"Local paper.md not found at {local_paper_path}, skipping verification")
            return True  # Skip verification if no local paper.md
        
        # Read local paper.md content
        local_content = local_paper_path.read_text()
        
        # Run a temporary container to extract the paper.md file
        container_name = f"verify-{task_id}-{os.getpid()}"
        
        try:
            # Start container
            run_cmd = [
                "docker", "run", "-d", "--name", container_name,
                "--platform", BEAKER_PLATFORM,
                beaker_image,
                "sh", "-c", "trap 'exit 0' TERM INT; while true; do sleep 1; done"
            ]
            subprocess.run(run_cmd, check=True, capture_output=True, text=True)
            
            # Extract paper.md from container
            extract_cmd = ["docker", "cp", f"{container_name}:/workspace/paper.md", "-"]
            result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
            container_content = result.stdout
            
            # Compare contents
            if local_content.strip() == container_content.strip():
                logger.info(f"Verification passed: paper.md files match for task {task_id}")
                return True
            else:
                logger.error(f"Verification failed: paper.md files differ for task {task_id}")
                logger.error(f"Local file size: {len(local_content)} bytes")
                logger.error(f"Container file size: {len(container_content)} bytes")
                return False
                
        finally:
            # Clean up container
            try:
                subprocess.run(["docker", "rm", "-f", container_name], 
                             check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to clean up container {container_name}")
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Verification failed for task {task_id}: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Verification error for task {task_id}: {e}")
        return False


def verify_task_wrapper(task_id, tasks_dir, workspace):
    """Wrapper function for multiprocessing that handles exceptions and logging."""
    try:
        return verify_task(task_id, tasks_dir, workspace)
    except Exception as e:
        # Return the error message instead of raising
        return str(e)


def verify_task(task_id, tasks_dir, workspace):
    """Verify a single task by checking container files match local files."""
    from minienv.examples.tbench.build_images import BeakerImagePusher
    
    pusher = BeakerImagePusher(workspace)
    image_name = get_image_name(task_id)
    beaker_image = f"{BEAKER_USER}/{image_name}"
    
    # Check if image exists on Beaker
    if not pusher.image_exists_on_beaker(image_name):
        logger.warning(f"Image for '{task_id}' does not exist on Beaker: '{beaker_image}'. Skipping...")
        return
    
    # Verify the container has the correct files
    logger.info(f"Verifying container files for task {task_id}...")
    verification_passed = verify_container_files(beaker_image, task_id, tasks_dir)
    if not verification_passed:
        raise RuntimeError(f"Verification failed for task {task_id}: files do not match")
    logger.info(f"✅ Verification completed successfully for task {task_id}")


def verify_tasks(tasks, tasks_dir, workspace, n_concurrent=None):
    """Verify tasks by checking container files match local files."""
    # Create a partial function with fixed arguments
    verify_task_with_args = partial(
        verify_task_wrapper, tasks_dir=tasks_dir, workspace=workspace
    )

    # Use process pool with configurable workers (default to 5 for verification)
    if n_concurrent is None:
        max_workers = min(5, len(tasks))
    else:
        max_workers = min(n_concurrent, len(tasks))
    logger.info(f"Verifying {len(tasks)} tasks using {max_workers} parallel workers")

    failed_tasks = []
    with Pool(processes=max_workers) as pool:
        # Map tasks to the process pool
        results = pool.map(verify_task_with_args, tasks)

        # Process results
        for task_id, result in zip(tasks, results):
            if result is None:
                # Success
                logger.info(f"Successfully verified task: {task_id}")
            else:
                # Failure - result contains the error message
                logger.error(f"\033[31mFailed to verify task {task_id}: {result}\033[0m")
                failed_tasks.append(task_id)

    return failed_tasks


def main(tasks, tasks_dir, workspace, n_concurrent=None):
    """Main function for verification."""
    # Ensure all tasks exist and are valid
    if tasks:
        tasks_to_verify = [tasks]
    else:
        tasks_path = Path(tasks_dir)
        if not tasks_path.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        tasks_to_verify = [
            item.name
            for item in tasks_path.iterdir()
            if item.is_dir() and (item / "docker-compose.yaml").exists()
        ]

        if not tasks_to_verify:
            logger.warning(f"No tasks found in {tasks_dir}")
            return

    # Verify tasks
    failed_tasks = verify_tasks(
        tasks=tasks_to_verify, tasks_dir=tasks_dir, workspace=workspace, n_concurrent=n_concurrent
    )

    if failed_tasks:
        raise RuntimeError(f"Failed verification tasks: {', '.join(failed_tasks)}")

    logger.info("All task verifications completed!")


def cli():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Verify T-Bench task containers")
    parser.add_argument("--task", type=str, help="Specific task to verify")
    parser.add_argument("--tasks-dir", type=str, default="tasks", help="Directory containing tasks")
    parser.add_argument("--workspace", type=str, default="ai2/rollouts", help="Beaker workspace")
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=None,
        help="Number of concurrent workers (default: 5 for verification)",
    )

    args = parser.parse_args()
    main(args.task, args.tasks_dir, args.workspace, args.n_concurrent)


if __name__ == "__main__":
    cli()
