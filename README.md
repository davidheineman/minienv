A docker service for LLM environments. Works locally with Docker or remote with Beaker.

### Setup

```sh
git clone https://github.com/davidheineman/minienv
pip install -e ".[all]"

# Ensure you daemon is summoned
docker run hello-world
```

### CLI Usage

`minienv` has a built-in CLI agent. Here's demo [`fibonacci`](./tasks/fibonacci/) and [`fre`](./tasks/fre/) (PaperBench) tasks:

```sh
# Show some example tasks
minienv list

# Run with local backend
minienv fibonacci

# Run with beaker backend
minienv fibonacci -b beaker

# Run an example from PaperBench
minienv fre
minienv fre -b beaker
```

### Python Usage

`minienv` can be used as a backend in Python:

```python
from minienv.backend import BeakerBackend

backend = BeakerBackend(
    workspace="ai2/rollouts"
)

await backend.create_env(
    task_name="test-task", 
    image="python:3.11-slim"
)

stdout, stderr, exit_code = \
    await backend.exec_command("ls", timeout=10)

print(stdout)

await backend.teardown()
```

### SWE Bench Example

Example of `minienv` on SWE Bench:

```sh
# Run a single instance with mini SWE agent!
pip install -e ".[swebench]"
python minienv/examples/swebench/swebench_single.py -i sqlfluff__sqlfluff-1625

# For full SWE bench, use their repo
python minienv/examples/swebench/swebench.py -w 30
```

<img width="650" alt="example" src="https://github.com/user-attachments/assets/2a1466d6-7d15-4ed1-b229-9ef914d5a4e8" />


