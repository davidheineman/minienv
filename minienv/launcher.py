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

def launch_beaker_job(name, description, docker_image, result_path='/results', workspace="ai2/davidh"):
    with Beaker.from_env() as beaker:
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


launch_beaker_job(
    name="davidh-awesome-job",
    description="awesome-job-by-davidh",
    docker_image="python:3.11-slim"
)