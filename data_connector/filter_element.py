import logging
from typing import List, Dict, Callable
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _no_op(data):
    return True


def _data_exists_by_key(key: str, data: Dict):
    has_key = key in data

    if not has_key:
        return False

    value = data[key]

    if value is None:
        return False

    if type(value) is list:
        return len(value) > 0
    else:
        return value is not None


def apply_filters(filters: List[Callable], data: Dict):
    for f in filters:
        result = f(data)
        if not result:
            return False

    return True


def get_filter_function(api: Dict) -> List[Callable]:
    key = "filter"

    if key not in api:
        return [_no_op]

    filter_configs = api[key]

    filter_funcs = []

    for config in filter_configs:
        if "dataExistsByKey" in config:
            key = config["dataExistsByKey"]
            func = partial(_data_exists_by_key, key)
            filter_funcs.append(func)
        else:
            raise RuntimeError("Unsupported filter configuration: {}".format(config))

    return filter_funcs
