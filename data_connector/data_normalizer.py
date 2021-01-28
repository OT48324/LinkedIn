import logging
from typing import List, Dict, Callable
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _concat_parent_child_keys(parent, child, separator='-'):
    if child:
        return parent + separator + child
    else:
        return parent


def _flatten_struct(struct, collected, skips) -> []:
    """
    Flattens the nested json structure. All the element in the `collected` list are tuple of key k and value v. Key k is
    the concatenated string value of all the keys traversed to get to the value v.  For example:
    {
        "a" : {
            "b" : "hello"
        },

        "c" : [
            {
                "d" : "world"
            },
            {
                "e" : "example"
            }
        ],

        "f" : {
            "g-h": "with dash"
        }
    }

    The above json would map into following json:

    [
     ("aB","hello"),
     ("c0D","world"),
     ("c1E","example"),
     ("fGH","with dash")
    ]

    Notice 0 in 'c0D'. That's the index information. Also f.g-h turned into 'fGH'.

    :param struct: Potential JSON object. This could be a list, object, or primitive
    :param collected: The final list containing key and value
    :param skips:
    :return:
    """
    if type(struct) is dict:
        for k, v in struct.items():
            if k not in skips:
                flattened = _flatten_struct(v, [], skips)
                new_acc = list(map(lambda x: (_concat_parent_child_keys(k, x[0]), x[1]), flattened))
                collected = collected + new_acc
    elif type(struct) is list:
        for i, elem in enumerate(struct):
            flattened = _flatten_struct(elem, [], skips)
            new_acc = list(map(lambda x: (_concat_parent_child_keys(str(i), x[0]), x[1]), flattened))
            collected = collected + new_acc
    else:
        collected.append(('', struct))

    return collected


def _list_to_dict(data):
    new_data = {}
    for k, v in data:
        new_data[k] = v
    return new_data


def _no_op(data):
    return data


def _flatten(data):
    logger.info("flattening..")
    normalized = []
    for d in data:
        flattened = _flatten_struct(d, [], ['links'])
        new_data = _list_to_dict(flattened)
        normalized.append(new_data)

    return normalized


def _get_normalize_func(agg_dict: Dict) -> Callable:
    if "flatten" in agg_dict:
        return partial(_flatten)
    else:
        raise RuntimeError("Unsupported normalizer configuration: {}".format(agg_dict))


def transform_dataset(api: Dict, data: List[Dict]) -> List[Dict]:
    if "datasetNormalizer" not in api:
        return _no_op(data)

    aggregators = api["datasetNormalizer"]

    for agg_dict in aggregators:
        logger.warning(
            "You are using aggregator function built in python. "
            "Can this be accomplished usign VSQL? We want to make sure transformations/aggregations are done through "
            "Vertica when possible.")
        agg_func = _get_normalize_func(agg_dict)
        data = agg_func(data)

    return data
