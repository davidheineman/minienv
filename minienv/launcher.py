import time
from beaker import Beaker, BeakerExperimentSpec, BeakerJobPriority


with Beaker.from_env() as beaker:
    # Build experiment spec...
    spec = BeakerExperimentSpec.new(
        budget="ai2/oe-eval",
        # workspace="ai2/davidh",
        description="beaker-py test run",
        beaker_image="petew/hello-world",
        priority=BeakerJobPriority.low,
        preemptible=True,
    )

    # Create experiment workload...
    workload = beaker.experiment.create(spec=spec)

    # Wait for job to be created...
    while (job := beaker.workload.get_latest_job(workload)) is None:
        print("waiting for job to start...")
        time.sleep(1.0)

    # Follow logs...
    print("Job logs:")
    for job_log in beaker.job.logs(job, follow=True):
        print(job_log.message.decode())