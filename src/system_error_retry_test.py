import os
import flyte
import flyte.errors

env = flyte.TaskEnvironment(
    name="system_error_retry",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse>=0.1.9"),
    reusable=flyte.ReusePolicy(
        replicas=(1, 2),
        idle_ttl=60,
        concurrency=10,
    ),
)

def get_attempt_number() -> int:
    return int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))

@env.task(retries=3)
async def system_failure_task(x: int) -> str:
    attempt = get_attempt_number()
    print(f"Task executing for input {x}, attempt number: {attempt}")

    if attempt < 2:
        print(f"Simulating system failure on attempt {attempt}")
        raise flyte.errors.RuntimeSystemError(
            "simulated_system_error",
            f"Simulated system failure on attempt {attempt}"
        )

    print(f"Success on attempt {attempt}!")
    return f"Succeeded on attempt {attempt}"

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(system_failure_task, x=42)
    print(run.url)
