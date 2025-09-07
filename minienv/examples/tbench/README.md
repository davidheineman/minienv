### setup

```sh
pip install terminal-bench
pip install minienv
pip install paramiko tenacity
```

### usage

```sh
# 1. Download tasks
python -m terminal_bench.cli.tb.datasets download --dataset terminal-bench-core==head

# (optional) specify an individual task
TASK_ID=hello-world

# 2. Build images
python build_images.py --task $TASK_ID

# 3. Evaluate task
python tb.py run \
    --task-id $TASK_ID \
    --agent claude-code \
    --model claude-sonnet-4-20250514 \
    --dataset-version head \
    --dataset-name terminal-bench-core
```

### todos
- [ ] add from docker compose:
    - [ ] volume mounts
    - [ ] environment keys
- [ ] there's an issue with image names including ".": `install-windows-3.11`
- [ ] some tasks (`tasks/security-celery-redis-rce`) use fancy docker-compose features that we can't support
