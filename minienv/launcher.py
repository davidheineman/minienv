import random
import string
import time
from beaker import Beaker, BeakerExperimentSpec, BeakerJobPriority
from rich.console import Console

console = Console()

AUS_CLUSTERS = [
    "ai2/saturn-cirrascale",
    "ai2/neptune-cirrascale",
    "ai2/jupiter-cirrascale-2",
    "ai2/ceres-cirrascale",
]

def launch_beaker_job(
    name, 
    description, 
    docker_image, 
    result_path='/results', 
    workspace="ai2/davidh"):
    beaker = Beaker.from_env()
    
    # random 4 char sequence
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))

    task_name = f"{name}-" + random_suffix
    
    spec = BeakerExperimentSpec.new(
        task_name=task_name,
        description=description,
        docker_image=docker_image,
        priority=BeakerJobPriority.normal,
        preemptible=True,
        budget="ai2/oe-eval",
        cluster=AUS_CLUSTERS,
        result_path=result_path

        # TODO: mount dataset
    )

    # Create beaker experiment
    try:
        with console.status("[bold green]creating beaker experiment...", spinner="dots") as status:
            workload = beaker.experiment.create(
                name=task_name,
                spec=spec, 
                workspace=workspace
            )
    except Exception as e:
        console.print(f"[bold red]Error creating beaker experiment: {e}[/bold red]")
        return

    # Wait for environment to initalize
    try:
        with console.status("[bold green]initializing beaker experiment...", spinner="dots") as status:
            while (job := beaker.workload.get_latest_job(workload)) is None:
                time.sleep(0.1)
        console.print("[bold green]environment setup complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing beaker experiment: {e}[/bold red]")
        return

    # Stream logs from the job
    try:
        for job_log in beaker.job.logs(job, follow=True):
            print(job_log.message.decode())
        console.print("[bold green]job completed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error streaming job logs: {e}[/bold red]")
        return


def create_dataset(
    dataset_name: str, 
    description: str,
    source_paths: list[str], 
    target_dir: str = None
    ):
    """Create beaker dataset"""
    beaker = Beaker.from_env()

    # Create beaker dataset
    try:
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
    except Exception as e:
        console.print(f"[bold red]Error creating dataset: {e}[/bold red]")
        raise
    
    # List uploaded files
    files = list(beaker.dataset.list_files(dataset))
    print(f"Seeint {len(files)} files:")
    for file in files:
        print(f"  - {file.path} ({file.size} bytes)")
    
    return dataset


create_dataset(
    dataset_name='my-dataset',
    description='this is a dataset',
    source_paths=['cli.py'],
)

launch_beaker_job(
    name="davidh-awesome-job",
    description="awesome-job-by-davidh",
    docker_image="python:3.11-slim"
)