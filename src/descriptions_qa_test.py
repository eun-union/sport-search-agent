"""
QA Test for Flyte Descriptions

This script tests three types of descriptions in Flyte 2.0:
1. TaskEnvironment descriptions
2. Task descriptions (short and long)
3. Input/output parameter descriptions

To run this test:
    python src/descriptions_qa_test.py
"""

import asyncio
import flyte

# ============================================================================
# Test 1: TaskEnvironment Description
# ============================================================================
# Description is passed as a parameter to the TaskEnvironment constructor

env_with_description = flyte.TaskEnvironment(
    name="qa-test-env",
    description="This is a test environment for validating description functionality in Flyte workflows",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base().with_pip_packages("requests"),
)


# ============================================================================
# Test 2: Task Descriptions (Short and Long)
# ============================================================================
# Task descriptions are automatically extracted from function docstrings
# - Short description: First line of the docstring
# - Long description: Remaining lines of the docstring

@env_with_description.task()
async def task_with_short_description() -> str:
    """This is a short description on a single line."""
    return "Task with short description completed"


@env_with_description.task()
async def task_with_long_description() -> str:
    """This is a short description (first line).

    This is the long description that provides more detailed information
    about what this task does, how it works, and any important notes.

    It can span multiple paragraphs and provide extensive documentation
    for developers who need to understand the task's behavior.
    """
    return "Task with long description completed"


# ============================================================================
# Test 3: Input/Output Descriptions
# ============================================================================
# Input and output descriptions are extracted from docstrings using standard
# Python documentation formats (Google, NumPy, or reStructuredText style)
# Below we use Google-style docstrings (Args/Returns sections)

@env_with_description.task()
async def calculate_discount(
    base_price: float,
    discount_percent: float,
    customer_tier: str = "standard",
) -> float:
    """Calculate the final price after applying a discount.

    This task demonstrates how to document input parameters and return values
    using the Google-style docstring format. The Flyte SDK automatically
    parses these descriptions and makes them available in the UI.

    Args:
        base_price: The original price of the item before any discounts
        discount_percent: The percentage discount to apply (0-100)
        customer_tier: The customer tier for additional discounts (standard, premium, vip)

    Returns:
        The final discounted price after applying all discounts
    """
    # Apply base discount
    discount_multiplier = 1.0 - (discount_percent / 100.0)
    discounted_price = base_price * discount_multiplier

    # Apply tier-based additional discount
    tier_discounts = {
        "standard": 0.0,
        "premium": 0.05,
        "vip": 0.10,
    }
    tier_discount = tier_discounts.get(customer_tier, 0.0)
    final_price = discounted_price * (1.0 - tier_discount)

    return final_price


@env_with_description.task()
async def process_order(
    order_id: str,
    items: list[str],
    quantities: list[int],
) -> dict[str, int | float | str]:
    """Process a customer order and calculate totals.

    This task takes order information and returns a summary with total
    item count and a processing status message.

    Args:
        order_id: Unique identifier for the order
        items: List of item names in the order
        quantities: List of quantities for each item (must match items length)

    Returns:
        A dictionary containing:
        - total_items: Total number of items ordered
        - order_id: The order identifier
        - status: Processing status message
    """
    if len(items) != len(quantities):
        raise ValueError("Items and quantities lists must have the same length")

    total_items = sum(quantities)

    return {
        "total_items": total_items,
        "order_id": order_id,
        "status": f"Processed {len(items)} unique items with total quantity {total_items}",
    }


# ============================================================================
# Test 4: NumPy-style docstring (alternative format)
# ============================================================================

@env_with_description.task()
async def analyze_metrics(
    metric_values: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Analyze a list of metric values and compute statistics.

    This demonstrates the NumPy-style docstring format, which is also
    supported by the Flyte SDK's docstring parser.

    Parameters
    ----------
    metric_values : list[float]
        A list of numerical metric values to analyze
    threshold : float, optional
        Threshold value for filtering outliers (default is 0.5)

    Returns
    -------
    dict[str, float]
        A dictionary containing:
        - mean: The average of the metric values
        - max: The maximum value
        - min: The minimum value
        - above_threshold: Count of values above the threshold
    """
    if not metric_values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "above_threshold": 0}

    mean_value = sum(metric_values) / len(metric_values)
    max_value = max(metric_values)
    min_value = min(metric_values)
    above_threshold = sum(1 for v in metric_values if v > threshold)

    return {
        "mean": mean_value,
        "max": max_value,
        "min": min_value,
        "above_threshold": float(above_threshold),
    }


# ============================================================================
# Main workflow that calls all test tasks
# ============================================================================

@env_with_description.task()
async def main_qa_workflow() -> dict[str, str]:
    """Main QA workflow that exercises all description types.

    This workflow calls all the test tasks to validate that descriptions
    are properly captured and processed by the Flyte SDK.

    Returns:
        A summary dictionary with results from all test tasks
    """
    # Test simple tasks
    result1 = await task_with_short_description()
    result2 = await task_with_long_description()

    # Test task with input/output descriptions
    discount_result = await calculate_discount(
        base_price=100.0,
        discount_percent=20.0,
        customer_tier="premium"
    )

    # Test task with multiple inputs and dict output
    order_result = await process_order(
        order_id="ORD-12345",
        items=["laptop", "mouse", "keyboard"],
        quantities=[1, 2, 1]
    )

    # Test task with NumPy-style docstring
    metrics_result = await analyze_metrics(
        metric_values=[0.1, 0.5, 0.8, 0.3, 0.9, 0.6],
        threshold=0.5
    )

    return {
        "status": "All QA tests completed successfully",
        "simple_task_1": result1,
        "simple_task_2": result2,
        "discount_result": f"Final price: ${discount_result:.2f}",
        "order_status": order_result["status"],
        "metrics_mean": f"{metrics_result['mean']:.2f}",
    }


if __name__ == "__main__":

    # Initialize Flyte from config
    flyte.init_from_config()

    # Run the main workflow
    run = flyte.run(main_qa_workflow)
    print(f"Run Name: {run.name}")
    print(f"Run URL: {run.url}")
