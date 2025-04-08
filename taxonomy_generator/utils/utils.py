import functools
import pickle
import random
from pathlib import Path
from typing import Callable, ParamSpec, Sequence, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
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


def cap_words(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split(" "))


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
    if seed is not None:
        random.seed(seed)
    return random.sample(population, min(n, len(population)))
