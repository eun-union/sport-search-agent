"""
Simple demo: t1->t2->t3 workflow where tasks can fail but downstream tasks continue

Key: Catch errors and provide fallback values so downstream tasks can run.
"""

import asyncio
import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    name="simple-continue",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)


@env.task
async def t1(should_fail: bool = False) -> int:
    if should_fail:
        raise ValueError("T1 failed!")
    return 10


@env.task
async def t2(value: int) -> int:
    return value * 2


@env.task
async def t3(value: int) -> str:
    return f"Final: {value}"


@env.task
async def workflow(t1_fails: bool = False) -> str:
    """Run t1->t2->t3 where t1 can fail but t2 and t3 still run"""

    # Run t1
    try:
        t1_result = await t1(should_fail=t1_fails)
        t1_status = "SUCCESS"
    except flyte.errors.RuntimeUserError:
        t1_result = 0
        t1_status = "FAILED"

    # Run t2 and t3
    t2_result = await t2(value=t1_result)
    t3_result = await t3(value=t2_result)

    return t3_result


if __name__ == "__main__":
    flyte.init_from_config()

    # Example: t1 fails, but t2 and t3 still run
    run = flyte.run(workflow, t1_fails=True)
    print(f"\nRun: {run.name}")
    print(f"URL: {run.url}")
