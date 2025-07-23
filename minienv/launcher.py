import random
import string
import time
from beaker import Beaker, BeakerDataMount, BeakerExperimentSpec, BeakerJobPriority
from rich.console import Console

from beaker.types import BeakerDataset

console = Console()

AUS_CLUSTERS = [
    "ai2/saturn-cirrascale",
    "ai2/neptune-cirrascale",
    "ai2/jupiter-cirrascale-2",
    "ai2/ceres-cirrascale",
]

def get_rand_suffix(k):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

def launch_beaker_job(
    name, 
    description, 
    docker_image, 
    result_path='/results', 
    workspace="ai2/davidh",
    mount: BeakerDataset = None
):
    beaker = Beaker.from_env()

    task_name = f"{name}-" + get_rand_suffix(k=4)

    CMD = 'ls /task'
    
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
                beaker=mount.id,
                mount_path='/task',
            )
        ],
        command=CMD.split(' ')
    )

    # Create beaker experiment
    with console.status("[bold green]creating beaker experiment...", spinner="dots") as status:
        workload = beaker.experiment.create(
            name=task_name,
            spec=spec, 
            workspace=workspace
        )

    # Wait for environment to initalize
    with console.status("[bold green]initializing beaker experiment...", spinner="dots") as status:
        while (job := beaker.workload.get_latest_job(workload)) is None:
            time.sleep(0.1)
    console.print("[bold green]environment setup complete![/bold green]")

    # Stream logs from the job
    for job_log in beaker.job.logs(job, follow=True):
        print(job_log.message.decode())
    console.print("[bold green]job completed![/bold green]")


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
    with console.status(f"[bold green]Creating dataset '{dataset_name}'...[/bold green]", spinner="dots") as status:
        dataset = beaker.dataset.create(
            dataset_name,
            *source_paths,
            target=target_dir,
            description=description,
            force=False,
            commit=True,
            strip_paths=False,
        )
    console.print(f"[bold green]dataset upload complete: {beaker.dataset.url(dataset)}[/bold green]")
    
    # Print uploaded files
    files = list(beaker.dataset.list_files(dataset))
    for file in files:
        print(f" - {file.path} ({file.size} bytes)")
    
    return dataset


dataset = create_dataset(
    name='my-dataset',
    description='this is a dataset',
    source_paths=['cli.py'],
)

launch_beaker_job(
    name="davidh-awesome-job",
    description="awesome-job-by-davidh",
    docker_image="python:3.11-slim",
    mount=dataset
)