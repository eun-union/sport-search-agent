import flyte

env = flyte.TaskEnvironment(
    name="retry_test",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    image=flyte.Image.from_debian_base(),
)

@env.task(retries=3)
async def flaky_task(x: int) -> str:
    print(f"Task executing for input {x}")
    print("Simulating failure...")
    raise RuntimeError(f"Simulated failure for input {x}")

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(flaky_task, x=42)
    print(run.url)
