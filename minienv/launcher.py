import random
import string
import time
from beaker import Beaker, BeakerDataMount, BeakerEnvVar, BeakerExperimentSpec, BeakerJobPriority, BeakerJob, BeakerWorkloadStatus
from rich.console import Console

from beaker.types import BeakerDataset

import requests
import json

console = Console()

AUS_CLUSTERS = [
    "ai2/saturn-cirrascale",
    "ai2/neptune-cirrascale",
    "ai2/jupiter-cirrascale-2",
    "ai2/ceres-cirrascale",
]

ENTRYPOINT = ['bash', '-c', 'python /server/server/main.py']

def get_rand_suffix(k):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

def launch_beaker_job(
    name, 
    description, 
    docker_image, 
    server_mount: BeakerDataset,
    task_mount: BeakerDataset,
    port,
    result_path='/results', 
    workspace="ai2/davidh",
) -> BeakerJob:
    beaker = Beaker.from_env()

    task_name = f"{name}-" + get_rand_suffix(k=4)

    spec = BeakerExperimentSpec.new(
        task_name=task_name,
        description=description,
        docker_image=docker_image,
        priority=BeakerJobPriority.normal,
        preemptible=True,
        budget="ai2/oe-eval",
        cluster=AUS_CLUSTERS,
        result_path=result_path,
        datasets=[
            BeakerDataMount.new(
                beaker=server_mount.id,
                mount_path='/server',
            ),
            BeakerDataMount.new(
                beaker=task_mount.id,
                mount_path='/task',
            )
        ],
        env_vars=[
            BeakerEnvVar(
                name="PYTHONUNBUFFERED",
                value=str(1)
            ),
            BeakerEnvVar(
                name="MINIENV_PORT",
                value=str(port)
            )
        ],
        command=ENTRYPOINT,
        
        # @davidh -- Careful with networking
        host_networking=True,
        arguments=["--port", port]
    )

    # Create beaker experiment
    with console.status("[bold yellow]creating beaker experiment...", spinner="dots") as status:
        workload = beaker.experiment.create(
            name=task_name,
            spec=spec, 
            workspace=workspace
        )

    # Wait for environment to initalize
    with console.status("[bold yellow]initializing beaker experiment...", spinner="dots") as status:
        while (job := beaker.workload.get_latest_job(workload)) is None:
            time.sleep(0.1)
    console.print("[bold green]environment setup complete![/bold green]")

    # Wait for startup
    with console.status("[bold yellow]waiting for job to start...", spinner="dots") as status:
        while job.status.status in [BeakerWorkloadStatus.submitted, BeakerWorkloadStatus.queued, BeakerWorkloadStatus.initializing]:
            time.sleep(0.1)
            job = beaker.workload.get_latest_job(workload)
            if job is None:
                raise RuntimeError("beaker job failed to start")
    console.print("[bold green]job started![/bold green]")

    return job


def create_dataset(
    name: str, 
    description: str,
    source_paths: list[str], 
    target_dir: str = None
    ):
    """Create beaker dataset"""
    beaker = Beaker.from_env()

    dataset_name = f"{name}-" + get_rand_suffix(k=4)

    # Create beaker dataset
    with console.status(f"[bold yellow]Creating dataset '{dataset_name}'...[/bold yellow]", spinner="dots") as status:
        dataset = beaker.dataset.create(
            dataset_name,
            *source_paths,
            target=target_dir,
            description=description,
            force=False,
            commit=True,
            strip_paths=False,
        )
    console.print(f"[bold green]dataset upload complete:[/bold green] {beaker.dataset.url(dataset)}")
    
    # Print uploaded files
    files = list(beaker.dataset.list_files(dataset))
    for file in files:
        print(f" - {file.path} ({file.size} bytes)")
    
    return dataset


def get_hostname(job: BeakerJob):
    beaker = Beaker.from_env()
    node_id = job.assignment_details.node_id
    node = beaker.node.get(node_id)
    hostname = node.hostname
    return hostname


def ping_server(hostname: str, port: int, timeout: int = 300):
    """Ping server until it responds or timeout is reached"""
    url = f"http://{hostname}:{port}/ping"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    return False

def exec_command(command: list[str], hostname: str, port: int) -> str:
    url = f"http://{hostname}:{port}/exec"
    headers = {"Content-Type": "application/json"}
    data = {"command": command}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise RuntimeError(f"Command execution failed: {response.status_code} {response.text}")
    response = response.json()
    response = response["stdout"]
    return response

def shutdown_server(hostname: str, port: int) -> bool:
    url = f"http://{hostname}:{port}/shutdown"
    response = requests.post(url)
    if response.status_code != 200:
        raise RuntimeError(f"Command execution failed: {response.status_code} {response.text}")
    response = response.json()
    if response["status"] == "shutting down":
        return True
    return False



def create_beaker_env(task_name: str, image: str):
    port = random.randint(1000, 10_000)

    server_dataset = create_dataset(
        name=f"minienv.{task_name}.server",
        description='server entrypoint',
        source_paths=['server/main.py'],
    )

    task_dataset = create_dataset(
        name=f"minienv.{task_name}.task",
        description='task files',
        source_paths=['/Users/dhei/ai2/paperbench/mini-envs/tasks/fibonacci'],
    )

    job: BeakerJob = launch_beaker_job(
        name=f"minienv.{task_name}",
        server_mount=server_dataset,
        task_mount=task_dataset,
        description=f"A minienv rollout: '{task_name}' on '{image}'",
        docker_image=image,
        port = port
    )

    hostname = get_hostname(job)

    # Wait for server to be ready
    with console.status("[bold yellow]waiting for server to be ready...", spinner="dots") as status:
        if not ping_server(hostname=hostname, port=port):
            console.print("[bold red]server failed to start within timeout![/bold red]")
            exit(1)
    console.print("[bold green]server is ready![/bold green]")

    ### you have to be on VPN to do this, but should add private key for extra layer of security. generate private/public key on execution

    return hostname, port


hostname, port = create_beaker_env(
    task_name = "fibonacci",
    image = "python:3.11-slim"
)

stdout = exec_command(command=['ls'], hostname=hostname, port=port)

print(stdout)

is_success = shutdown_server(hostname=hostname, port=port)

if is_success:
    console.print("[bold green]Teardown successful![/bold green]")