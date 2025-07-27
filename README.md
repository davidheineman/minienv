A docker service for LLM environments. Each rollout has access to its own container.

### Setup

```sh
pip install -e ".[all]"

# Ensure you daemon is summoned
docker run hello-world
```

### CLI Usage

Here's an example with a demo `fibonacci` task:

```sh
# Show some example tasks
minienv list

# Run with local backend
minienv fibonacci

# Run with beaker backend
minienv fibonacci -b beaker
```

### Python Usage

You can use `minienv` as a backend in Python:

```python
from minienv.backend import BeakerBackend

backend = BeakerBackend("ai2/rollouts")

await backend.create_env(task_name="test-task", image="python:3.11-slim")

stdout, stderr, exit_code = await backend.exec_command(["ls"], timeout=10)

print(stdout)

await backend.teardown()
```

### SWE Bench Example

Example of `minienv` on SWE Bench:

```sh
# Run a single instance with mini SWE agent!
pip install mini-swe-agent
python minienv/examples/swebench/swebench_single.py -i sqlfluff__sqlfluff-1625

# For full SWE bench, use their repo
python src/minisweagent/run/extra/swebench.py -w 30 --redo-existing
```

<!-- python src/minisweagent/run/extra/swebench_single.py -i sqlfluff__sqlfluff-1625 -->