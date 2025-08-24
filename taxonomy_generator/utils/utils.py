import functools
import json
import pickle
import random
import signal
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Literal, ParamSpec, TypeVar, cast, overload

import matplotlib.pyplot as plt
from pydantic import BaseModel

T = TypeVar("T")
R = TypeVar("R")
D = TypeVar("D")
P = ParamSpec("P")
GR = TypeVar("GR", bound=Generator[Any, Any, Any])
DI = TypeVar("DI", bound=dict[str, Any])


class Recursor(BaseModel):
    gen: Generator[None, Any]
    depth: int
    complete: bool


@contextmanager
def timeout(seconds: int, operation_descriptor: str = "Operation"):
    def timeout_handler(signum: int, frame: FrameType | None):  # pyright: ignore[reportUnusedParameter]
        raise TimeoutError(f"{operation_descriptor} timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def recurse_even(func: Callable[P, GR]):
    """Makes a generator function recursive, while ensureing the function is evaluated one depth at a time."""

    def wrapper(max_depth: int = 3, *args: P.args, **kwargs: P.kwargs):
        recursors: list[Recursor] = []

        def call_child(
            _current_depth: int,
            *child_args: P.args,
            **child_kwargs: P.kwargs,
        ):
            def this_call_child(*args: P.args, **kwargs: P.kwargs):
                call_child(_current_depth + 1, *args, **kwargs)

            generator = func(this_call_child, *child_args, **child_kwargs)  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
            recursors.append(
                Recursor(
                    gen=generator,  # pyright: ignore[reportUnknownArgumentType]
                    depth=_current_depth,
                    complete=(next(generator, -1) == -1),  # pyright: ignore[reportUnknownArgumentType]
                )
            )

        depth = 0
        call_child(depth, *args, **kwargs)

        while depth < max_depth and any((not r.complete) for r in recursors):
            while any((not r.complete) for r in recursors if r.depth == depth):
                for r in recursors:
                    if r.depth == depth:
                        r.complete = next(r.gen, -1) == -1

            depth += 1

    return wrapper


def cache(cache_filename_override: str | None = None, max_size: int | None = 30):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache_dir = Path(".func_cache")
        cache_dir.mkdir(exist_ok=True)

        cache_file = cache_dir / f"{cache_filename_override or func.__name__}.pkl"
        # Load cache if exists
        if cache_file.exists():
            with cache_file.open("rb") as f:
                cache = pickle.load(f)  # noqa: S301
        else:
            cache: dict[str, Any] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
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

        return wrapper  # type: ignore

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
        text,
        lambda w: w if any(char.isupper() for char in w) else w.capitalize(),
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
    population: Sequence[T],
    n: int = 1,
    seed: int | None = None,
) -> list[T]:
    random.seed(seed)
    return random.sample(population, min(n, len(population)))


def get_avg_deviation(nums: list[int]) -> float:
    mean = sum(nums) / len(nums)
    deviations = [abs(num - mean) for num in nums]
    mean_deviation = sum(deviations) / len(deviations)
    return mean_deviation / mean


def unique_str(only_date: bool = False) -> str:
    if only_date:
        return time.strftime("%Y-%m-%d")
    return f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(10000, 99999)}"


def plot_list(
    data: list[float] | dict[int, float],
    title: str = "Results",
    kind: Literal["line", "bar"] = "line",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
):
    if isinstance(data, dict):
        x = list(data.keys())
        y = list(data.values())
    else:
        x = list(range(1, len(data) + 1))
        y = data

    match kind:
        case "line":
            plt.plot(  # pyright: ignore[reportUnknownMemberType]
                x,
                y,
                marker="o",  # marker='o' for dots
            )
        case "bar":
            plt.bar(x, y)  # pyright: ignore[reportUnknownMemberType]

    plt.xlabel(xlabel)  # pyright: ignore[reportUnknownMemberType]
    plt.ylabel(ylabel)  # pyright: ignore[reportUnknownMemberType]
    plt.title(title)  # pyright: ignore[reportUnknownMemberType]

    plt.xticks(x)  # pyright: ignore[reportUnknownMemberType]

    plt.show()  # pyright: ignore[reportUnknownMemberType]


def compare_datas(
    data1: dict[int, float],
    data2: dict[int, float],
    ylabel: str,
    data1_label: str,
    data2_label: str,
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # pyright: ignore[reportUnknownMemberType]
    ax1.bar(data1.keys(), data1.values())
    ax1.set_title(data1_label)
    ax1.set_ylabel(ylabel)
    ax2.bar(data2.keys(), data2.values())
    ax2.set_title(data2_label)
    ax2.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]

    # Plot on same graph
    plt.figure(figsize=(10, 6))  # pyright: ignore[reportUnknownMemberType]
    plt.bar(list(data1.keys()), list(data1.values()), alpha=0.7, label=data1_label)  # pyright: ignore[reportUnknownMemberType]
    plt.bar(list(data2.keys()), list(data2.values()), alpha=0.7, label=data2_label)  # pyright: ignore[reportUnknownMemberType]
    plt.legend()  # pyright: ignore[reportUnknownMemberType]
    plt.title("Comparison")  # pyright: ignore[reportUnknownMemberType]
    plt.xlabel("Number of Topics")  # pyright: ignore[reportUnknownMemberType]
    plt.ylabel(ylabel)  # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]


@overload
def normalize(data: list[float], out_of: float = 100) -> list[float]: ...


@overload
def normalize(data: dict[T, float], out_of: float = 100) -> dict[T, float]: ...


def normalize(
    data: list[float] | dict[T, float], out_of: float = 100
) -> list[float] | dict[T, float]:
    if isinstance(data, list):
        total = sum(data)
        return [v / total * out_of for v in data]

    total = sum(data.values())
    return {k: v / total * out_of for k, v in data.items()}


@overload
def switch(
    var: T,
    options: Iterable[tuple[T, R]],
    default: None = None,
) -> R | None: ...


@overload
def switch(var: T, options: Iterable[tuple[T, R]], default: D) -> R | D: ...


def switch(var: T, options: Iterable[tuple[T, R]], default: object | None = None):
    return next((value for option, value in options if option == var), default)


def resolve_all_param(
    param: T | Sequence[T],
    idx: int,
    iter_type: type[Sequence[Any]] = list,
) -> T:
    return (  # pyright: ignore[reportReturnType,reportUnknownVariableType]
        (param[idx] if 0 <= idx < len(param) else param[-1])  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(param, iter_type)
        else param
    )


def get_resolve_all_param(idx: int, iter_type: type[Sequence[Any]] = list):
    def resolver(param: T | Sequence[T]) -> T:
        return resolve_all_param(param, idx, iter_type)

    return resolver


def join_items_english(items: list[str]) -> str:
    meta_str = ""
    for i in range(len(items) - 1):
        if i < len(items) - 2:
            meta_str += items[i] + ", "
        else:
            meta_str += items[i] + " and "

    return meta_str + items[-1]


def save_pydantic(model: BaseModel, path: Path, indent: int | None = 2):
    path.write_text(
        json.dumps(
            model.model_dump(exclude_defaults=True),
            ensure_ascii=False,
            indent=indent,
        )
    )


def clean_dict(d: DI) -> DI:
    return cast(DI, {k: v for k, v in d.items() if v is not None})
