A docker service for LLM environments. Each rollout has access to its own container.

```sh
# Ensure you daemon is summoned
docker run hello-world
```

```sh
python runner.py list
```

```sh
python runner.py hello_world
python runner.py fibonacci
python runner.py attention
```