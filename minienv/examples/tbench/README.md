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
    # swebench-verified==head
    # mlebench-lite==head
    # sweperf==head
    # swesmith==head
tb datasets download --dataset terminal-bench-core==0.1.1

# Build images
python minienv/examples/tbench/build_images.py \
    --tasks-dir /root/.cache/terminal-bench/terminal-bench-core/0.1.1

# Evaluate on Terminal-Bench
alias minitb="python minienv/examples/tbench/tb.py"

minitb run \
    -a oracle \
    -d terminal-bench-core==0.1.1 \
    --n-concurrent 30

minitb run \
    -a claude-code \
    -m claude-sonnet-4-20250514 \
    -a oracle \
    -d terminal-bench-core==0.1.1 \
    --n-concurrent 30
```

### todos
- [ ] randomize the hosts when launching jobs (it all defaults to the same host)
- [ ] builds are failing for images that specify platforms other than `linux/amd64`
- [ ] add from docker compose:
    - [ ] volume mounts
    - [ ] environment keys
    - [ ] arbitrary docker compose builds, with failures on unsupported configs
- [ ] there's an issue with image names including ".": `install-windows-3.11`
- [ ] some tasks (`tasks/security-celery-redis-rce`) use fancy docker-compose features that we can't easily support
