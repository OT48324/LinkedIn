import logging
from typing import List, Dict, Callable
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dedup(key: str, data: List[Dict]):
    return list({v[key]: v for v in data}.values())


def _no_op(data):
    return data


def _get_agg_func(agg_dict: Dict) -> Callable:
    if "dedupByKey" in agg_dict:
        return partial(_dedup, agg_dict["dedupByKey"])
    else:
        raise RuntimeError("Unsupported aggregator configuration: {}".format(agg_dict))


def aggregate_dataset(api: Dict, data: List[Dict]) -> List[Dict]:
    if "datasetAggregator" not in api:
        return _no_op(data)

    aggregators = api["datasetAggregator"]

    for agg_dict in aggregators:
        logger.warning(
            "You are using transformation function built in python. "
            "Can this be accomplished usign VSQL? We want to make sure transformations/aggregations are done through "
            "Vertica when possible.")
        agg_func = _get_agg_func(agg_dict)
        data = agg_func(data)

    return data
