A docker service for LLM environments. Each rollout has access to its own container.

```sh
pip install -e ".[all]"
```

```sh
# Ensure you daemon is summoned
docker run hello-world
```

```sh
minienv list
```

```sh
minienv hello_world
minienv fibonacci
minienv attention
```