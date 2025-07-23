import argparse
import asyncio

from minienv.runner import list_tasks_example, run_task_example


async def main():
    parser = argparse.ArgumentParser(description="MiniEnv - Run tasks in containerized environments")
    parser.add_argument("task", nargs="?", help="Task ID to run (or 'list' to list available tasks)")
    parser.add_argument("-b", "--backend", choices=["local", "beaker"], default="local", 
                       help="Backend to use (default: local)")
    parser.add_argument("--traces-dir", default="traces", 
                       help="Directory to save conversation traces (default: traces)")
    
    args = parser.parse_args()

    if args.task == "list" or args.task is None:
        await list_tasks_example()
    else:
        await run_task_example(args.task, backend=args.backend, traces_dir=args.traces_dir)


def cli_main():
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
