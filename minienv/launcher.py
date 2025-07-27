#!/usr/bin/env python3
"""
Simple launcher using BeakerBackend.
"""
import asyncio
from minienv.backend.beaker import BeakerBackend
from minienv.constants import TASKS_DIR

async def main():
    backend = BeakerBackend()

    await backend.create_env(task_name="fibonacci", image="python:3.11-slim", task_files = [TASKS_DIR / "fibonacci"])

    stdout, stderr, exit_code = await backend.exec_command("ls")

    print(f"stdout: {stdout}")
    print(f"stderr: {stderr}")
    print(f"exit_code: {exit_code}")

    success = await backend.teardown()
    print(f"Teardown successful: {success}")

if __name__ == "__main__":
    asyncio.run(main()) 