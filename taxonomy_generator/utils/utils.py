import functools
import pickle
import random
import time
from pathlib import Path
from typing import Callable, Iterable, ParamSpec, Sequence, TypeVar, overload

import matplotlib.pyplot as plt
from pydantic import BaseModel

T = TypeVar("T")
R = TypeVar("R")
D = TypeVar("D")
P = ParamSpec("P")


def cache(cache_filename_override: str | None = None, max_size: int | None = 30):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache_dir = Path(".func_cache")
        cache_dir.mkdir(exist_ok=True)

        cache_file = cache_dir / f"{cache_filename_override or func.__name__}.pkl"
        # Load cache if exists
        if cache_file.exists():
            with cache_file.open("rb") as f:
                cache = pickle.load(f)
        else:
            cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache
            # Create key from arguments
            key = str(args) + str(sorted(kwargs.items()))

            if key not in cache:
                # Compute and cache result
                cache[key] = func(*args, **kwargs)

                if max_size:
                    if len(cache) > max_size:
                        new_cache = {}
                        for i, (k, v) in enumerate(cache.items()):
                            if len(cache) - i <= max_size:
                                new_cache[k] = v
                        cache = new_cache

                # Save updated cache
                with cache_file.open("wb") as f:
                    pickle.dump(cache, f)

            return cache[key]

        return wrapper

    return decorator


def log(p: str):
    print(p + "\n\n")


def split_join(
    text: str,
    func: Callable[[str], str],
    seps: list[str] = [" ", "-"],
) -> str:
    if not seps:
        return func(text)

    current_sep = seps[0]
    remaining_seps = seps[1:]

    parts = text.split(current_sep)
    processed_parts = [split_join(part, func, remaining_seps) for part in parts]

    return current_sep.join(processed_parts)


def cap_words(text: str) -> str:
    return split_join(
        text, lambda w: w if any(char.isupper() for char in w) else w.capitalize()
    )


def safe_lower(text: str) -> str:
    return split_join(text, lambda w: w.lower() if w.capitalize() == w else w)


def serialize(obj: list[BaseModel] | BaseModel):
    return (
        obj.model_dump()
        if isinstance(obj, BaseModel)
        else [o.model_dump() for o in obj]
    )


def format_perc(value: float, fill: bool = False) -> str:
    rounded = round(value * 100, 2)
    if rounded % 1 == 0 and not fill:
        rounded = int(rounded)
    return f"{rounded:.2f}%" if fill else f"{rounded}%"


def random_sample(
    population: Sequence[T], n: int = 1, seed: int | None = None
) -> list[T]:
    random.seed(seed)
    return random.sample(population, min(n, len(population)))


def get_avg_deviation(nums: list[int]) -> float:
    mean = sum(nums) / len(nums)
    deviations = [abs(num - mean) for num in nums]
    mean_deviation = sum(deviations) / len(deviations)
    return mean_deviation / mean


def unique_str() -> str:
    return f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(10000, 99999)}"


def plot_list(arr: list[float], title="Results"):
    x = [i for i in range(1, len(arr) + 1)]

    # Create the plot
    plt.plot(x, arr, marker="o")  # Added marker='o' to show dots at each point

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)

    # Set x-axis ticks at each data point
    plt.xticks(x)

    # Display the plot
    plt.show()


@overload
def switch(
    var: T, options: Iterable[tuple[T, R]], default: None = None
) -> R | None: ...


@overload
def switch(var: T, options: Iterable[tuple[T, R]], default: D) -> R | D: ...


def switch(var: T, options: Iterable[tuple[T, R]], default: D | None = None):
    return next((value for option, value in options if option == var), default)
