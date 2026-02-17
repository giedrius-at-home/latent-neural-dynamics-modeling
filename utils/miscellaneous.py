from pathlib import Path
from typing import KeysView, Optional
from utils.logger import get_logger


def length(_list: list | None) -> int | None:
    if _list is None:
        return None
    return len(_list)


def flatten(nested_list: list[list]) -> list:
    logger = get_logger()
    if not isinstance(nested_list[0], list):
        logger.info("Input is not a nested list.")
        return nested_list
    flat_list = []
    for row in nested_list:
        flat_list += row
    return flat_list


def lookup_keys(keys: KeysView[str], lookup_keywords: list) -> tuple:

    fetched_keys = {
        key_
        for key_ in keys
        if any(lookup_keyword in key_ for lookup_keyword in lookup_keywords)
    }

    return tuple(fetched_keys)


def contains_nulls(lst: Optional[list]) -> bool:
    if lst is None:
        return True
    return any(x is None for x in lst)


def state_shape(state):
    return (
        [p.shape for p in state if p is not None]
        if isinstance(state, list)
        else state.shape
    )


def get_latest_timestamp(directory: str) -> str:
    models = Path(directory).glob("*.pkl")
    timestamps = [
        str(model.name).replace(".pkl", "").replace("model_", "") for model in models
    ]
    return max(timestamps)
