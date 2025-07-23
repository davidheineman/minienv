import asyncio

from minienv.runner import list_tasks_example, run_task_example


async def example_usage():
    import sys
    
    task_id = None
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
    
    if task_id == "list":
        await list_tasks_example()
    else:
        await run_task_example(task_id)


def main():
    asyncio.run(example_usage())


if __name__ == "__main__":
    main()