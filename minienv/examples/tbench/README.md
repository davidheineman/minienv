### setup

```sh
pip install "minienv[beaker,dev,docker,tbench]"
```

<details>
<summary>local install</summary>

```sh
git clone https://github.com/davidheineman/minienv.git
cd minienv
pip install -e ".[beaker,dev,docker,tbench]"
```

</details>


### usage

```sh
# List datasets
tb datasets list

# Download tasks
tb datasets download --dataset terminal-bench-core==head

# Build images
python minienv/examples/tbench/build_images.py \
    --tasks-dir /root/.cache/terminal-bench/terminal-bench-core/head

# Evaluate task
alias minitb="python minienv/examples/tbench/tb.py"
minitb run \
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
    - [ ] arbitrary docker compose builds, with failures on unsupported configs
- [ ] there's an issue with image names including ".": `install-windows-3.11`
- [ ] some tasks (`tasks/security-celery-redis-rce`) use fancy docker-compose features that we can't easily support
