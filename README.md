A docker service for LLM environments. Each rollout has access to its own container.

### Setup

```sh
pip install -e ".[all]"
```

```sh
# Ensure you daemon is summoned
docker run hello-world
```

### Usage

```sh
minienv list
```

```sh
# Run with local backend
minienv hello_world
minienv fibonacci
minienv attention

# Run with beaker backend
minienv hello_world -b beaker
minienv fibonacci -b beaker
minienv attention -b beaker
```