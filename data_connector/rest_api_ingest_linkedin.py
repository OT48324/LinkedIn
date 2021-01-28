import argparse
import datetime
import json
import logging
import os
import time
import urllib
from datetime import datetime, date, timedelta
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Union, List, Dict, Tuple, Callable, Any

import asyncio
import boto3
import pytz
import requests
import requests_async
import yaml
from botocore.exceptions import ClientError

import data_normalizer
#import data_aggregator
import filter_element

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ffi.rest_api")
timezone = pytz.timezone("US/Eastern")


def parse_args():
    """
    Argument parsing logic
    :return:
    """
    parser = argparse.ArgumentParser(
        description="This process pulls Ny Reg 187 data based on the specified configuration."
    )
    parser.add_argument(
        "--client-id", required=True,
        help="Client id of application."
    )
    parser.add_argument(
        "--client-secret", required=True,
        help="Client secret of application."
    )
    parser.add_argument(
        "--access-token", required=True,
        help="Client token of application."
    )
    parser.add_argument(
        "--default-start-date", required=True,
        help="Default first day to pull data for."
    )
    parser.add_argument(
        "--default-end-date", required=True,
        help="Default last day to pull data for."
    )
    parser.add_argument(
        "--conf-file", help="Configuration file for a data type (python module).",
        required=True
    )
    parser.add_argument(
        "--meta-date-format", help="Date format that is used to store historical metadata of already-ingested api.",
        required=True
    )
    parser.add_argument(
        "--s3-bucket", help="S3 Bucket where the metadata lies.",
        required=True
    )
    parser.add_argument(
        "--metadata-file-key", help="Key to the object in s3 containing the metadata",
        required=True
    )
    parser.add_argument(
        "--delta-day-range", help="Delta day range.",
        required=True
    )
    args = parser.parse_args()
    return args


def get_days(day: str) -> int:
    return int(day[:-4])


def get_minutes(day: str) -> int:
    return int(day[:-7])


def get_seconds(day: str) -> int:
    return int(day[:-7])


def get_days_delta(delta_range: str):
    return timedelta(days=get_days(delta_range))


def get_minutes_delta(delta_range: str):
    return timedelta(minutes=get_minutes(delta_range))


def get_seconds_delta(delta_range: str):
    return timedelta(seconds=get_seconds(delta_range))


def get_time_delta(delta_range: str):
    if delta_range.endswith("days"):
        return get_days_delta(delta_range)
    elif delta_range.endswith("seconds"):
        return get_seconds_delta(delta_range)
    elif delta_range.endswith("minutes"):
        return get_minutes_delta(delta_range)
    else:
        raise RuntimeError("Unsupported time range specified: {}".format(delta_range))


def get_range(delta_range: str) -> int:
    if delta_range.endswith("days"):
        return get_days(delta_range)
    elif delta_range.endswith("seconds"):
        return get_seconds(delta_range)
    elif delta_range.endswith("minutes"):
        return get_minutes(delta_range)
    else:
        raise RuntimeError("Unsupported time range specified: {}".format(delta_range))


def get_time_diff(end: date, start: date, time_range: str):
    if time_range.endswith("days"):
        return (end - start).days
    elif time_range.endswith("seconds"):
        return (end - start).seconds
    elif time_range.endswith("minutes"):
        return (end - start).min
    else:
        raise RuntimeError("Unsupported time range specified: {}".format(time_range))


def get_start_end_dates(start: date, end: date, time_range="7days") -> list:
    """
    It finds all the dates between start and end with a certain range.
    :param start: Start date. Ex: 2019-01,01
    :param end: End Date. Ex 2019-08-16
    :param time_range: 7
    :return: List of dates. Each element is a tuple (s,e) where Date e is no further than day_range than date s.
    """
    collected = []

    range_num = get_range(time_range)

    def find_dates(s):
        remaining = get_time_diff(end, s, time_range)
        # print("this is remaining {}".format(remaining))
        if 0 <= remaining <= range_num:
            collected.append((s, end))
        elif remaining < 0:
            raise RuntimeError(
                "End date cannot be greater than start date. End date = {}, Start date = {}".format(end, s))
        else:  # difference between start date and end date is more than 7 days

            # add 7 days to the start date, this is your end date for this pair
            new_end = s + get_time_delta(time_range)

            # store the pair
            collected.append((s, new_end))

            # start a new iteration where start date is this end
            find_dates(new_end)

    find_dates(start)

    return collected


def get_token(token_uri, client_id, secret, scope) -> str:
    """
    Gets a valid bearer token.
    :param token_uri: token url
    :param client_id: Client Id
    :param secret: Client Secret
    :param scope: Scope of the API Call
    :return: String token
    """

    request_body = dict(grant_type="client_credentials",
                        client_id=client_id,
                        client_secret=secret,
                        scope=scope)

    def request():
        response = requests.post(token_uri, data=request_body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
        return response

    resp = request()

    while resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        logger.info("Waiting 5 seconds because 10 API call per limit quota..")
        time.sleep(5)
        logger.info("Requesting again..")
        resp = request()

    if resp.status_code == HTTPStatus.OK:
        data = resp.json()
        token = "Bearer " + data["access_token"]
        return token
    else:
        raise RuntimeError("Can not get [Degreed Oauth2 Token],\n Token Service return: {}".format(resp.status_code))


async def fetch_json_auth2_async(full_url, access_token) -> Dict:
    """
    Makes the API Call
    :param full_url:
    :param access_token:
    :return:
    """

    async def request():
        
        oauth_token='&oauth2_access_token='
        full_url_1=full_url+oauth_token+access_token
        response = await requests_async.get(full_url_1)
        return response

    resp = await request()

    while resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        retry_after = int(resp.headers['retry-after'])
        logger.info("Waiting {} seconds because 10 API call per limit quota..".format(retry_after))
        time.sleep(retry_after)
        logger.info("Requesting again..")
        resp = await request()

    # if resp.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
    #     logger.error("Server error with status code {}.\n'{}\n{}\n Will skip and keep going for other apis".format(
    #         resp.status_code, resp.text,
    #         resp.reason))
    #     raise RuntimeError("Response error with status code {}.\n{}\n{}".format(resp.status_code, resp.text, resp.reason))

    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(
            "Response error with status code {}.\n{}\n{}".format(resp.status_code, resp.text, resp.reason))
    else:
        return resp.json()


def extract_from_data_key(response) -> Union[str, Dict]:
    """
    Function that returns value of "data" in the dictionary
    :param response: Dictionary
    """
    return response["data"]


def extract_no_op(response) -> Any:
    """
    Function that does nothing to the response and returns the same object
    :param response: Anything
    """
    return response


def output_file_path(api, app_run_time, target_data_path):
    """
    Given the api and the run time of the application,
    returns the output path of the file where this apis data is supposed to be stored.
    :param api: API. Please look at file configs/degreed_config.yaml.
    :param app_run_time: Time when the application was run.
    :param target_data_path: base output directory where all the downloaded artifacts exists
    :return: string path to the data
    """
    return target_data_path + "/" + api["target"].replace("????", app_run_time).strip("/")


def _contains_more_data(res) -> bool:
    """
    Returns true if the json response contains more data
    :param res:
    :return:
    """
    if "links" in res:
        more_data = [res["links"] is not None, "next" in res["links"]]
        return all(more_data)
    else:
        return False


def _get_extract_function(api: Dict):
    if "extractFromResponse" not in api:
        return extract_no_op

    name = api["extractFromResponse"]

    func_libray = {
        "extract_from_data_key": extract_from_data_key,
        "no_op": extract_no_op
    }

    if name in func_libray:
        return func_libray[name]
    else:
        raise RuntimeError("Unknown extract function name: {}".format(name))


async def fetch_json_to_df(full_url: str, access_token: str,
                           filter_funcs: List[Callable], extract_func: Callable,
                           get_next_page: Callable) -> Union[None, List]:
    """
    Fetches all the data.  It is possible for the api to send data in pages.
    :param filter_funcs: List of function that is applied to the data. This is to remove data based on the filter
    :param extract_func: a functions that maps a given json response to another one
    :param get_next_page:
    :param full_url:
    :param access_token:
    :return:
    """
    logger.info("Requesting API: '{}'".format(full_url))

    res = await fetch_json_auth2_async(full_url, access_token)

    if res is None:
        logger.info("Response is NONE for url: '{}'".format(full_url))
        return None

    extracted = extract_func(res)

    data = []

    if type(extracted) is list:
        filter_applicator = partial(filter_element.apply_filters, filter_funcs)
        filtered = [x for x in extracted if filter_applicator(x)]
        data.extend(filtered)
    else:
        result = filter_element.apply_filters(filter_funcs, extracted)
        if result:
            data.append(extracted)

    next_url = get_next_page(res)

    while next_url is not None:

        logger.info("More Data Available at: '{}'".format(next_url))

        res = await fetch_json_auth2_async(next_url, access_token)

        extracted = extract_func(res)

        if type(extracted) is list:
            data.extend(extracted)
        else:
            data.append(extracted)

        next_url = get_next_page(res)

    return data


def _write_to_file(path: str, data: Union[List[Dict], Dict]) -> None:
    """Write the data to file.
    Will overwrite existing file.
    :param path: Path to file to write to.
    :param data: Dict or List of Dict to write to the file.
    :return: None
    :raise TypeError if data is not List[Dict] or Dict
    """

    if isinstance(data, Dict):
        data = [data]
    if not isinstance(data, List):
        raise TypeError("type of data parameter must be List[Dict] or Dict.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w+") as f:
        for datum in data:
            line = json.dumps(datum)
            f.write(line + "\n")

    return None


def _set_time_param(start_date_query_key: str, end_date_query_key: str,
                    start: str, end: str) -> dict:
    """
    Set filter for start and end date
    :param start:
    :param end:
    :return:
    """
    return {start_date_query_key: start, end_date_query_key: end}


def _with_query_params(api: Dict, initial_params: Dict, base_uri: str) -> str:
    if "queryParams" in api:
        for x in api["queryParams"]:
            initial_params.update(x)

    if initial_params:
        return base_uri + api["path"] + "?" + urllib.parse.urlencode(initial_params)
    else:
        return base_uri + api["path"]


async def _process_api_with_timestamp(api: Dict, access_token: str, base_uri: str,
                                      all_dates: List[Tuple[datetime, datetime]], date_format: str,
                                      start_date_query_key: str, end_date_query_key: str,
                                      next_page_func: Callable) -> list:
    """
    Process all the api between start and end dates
    :param api: Api being processed
    :param access_token: Access_token for authentication to the api
    :param base_uri: Base url of the api. For example, "https://api.degreed.com/api/"
    :param all_dates: Some apis require to use timestamp in the request.
    Degreed api allow date range not more than 7 days. all_dates is a list of date ranges
    [(x_1,y_1),(x_2,y_2),...,(x_n,y_n)] where y-x <= 7 days.
    :return:
    """
    name = api["name"]

    collected = []

    for start, end in all_dates:

        start_norm = start.strftime(date_format)

        end_norm = end.strftime(date_format)

        path_param = _set_time_param(start_date_query_key, end_date_query_key, start_norm, end_norm)

        full_url = _with_query_params(api, path_param, base_uri)

        extract_func = _get_extract_function(api)

        filter_funcs = filter_element.get_filter_function(api)

        data = await fetch_json_to_df(full_url, access_token, filter_funcs, extract_func, next_page_func)

        if data is None:
            logger.info("Something went wrong with accessing  api: '{}'".format(full_url))
        elif len(data):
            fetched_set = {
                'start_date': start_norm,
                'end_date': end_norm,
                'dataset': data
            }
            logger.info(
                "Records retrieved between '{}' and '{}' for '{}' = '{}'".format(start_norm, end_norm, name, len(data)))
            collected.append(fetched_set)
        else:
            logger.info("Records retrieved between '{}' and '{}' for '{}' = '{}'".format(start_norm, end_norm, name, 0))

    return collected


async def _process_api_without_timestamp(api, access_token, base_uri, next_page_func) -> list:
    """
    Process the api without any time restriction
    :param api: Api being processed
    :param access_token: Access_token for authentication to the api
    :param base_uri: Base url of the api. For example, "https://api.degreed.com/api/"
    :return:
    """

    full_url = _with_query_params(api, {}, base_uri)

    extract_func = _get_extract_function(api)

    filter_funcs = filter_element.get_filter_function(api)

    collected = await fetch_json_to_df(full_url, access_token, filter_funcs, extract_func, next_page_func)

    return collected


async def _process_api(api: Dict, access_token: str, base_uri: str, all_dates: List[Tuple[datetime, datetime]],
                       app_run_time: str, target_data_path: str, date_format: str,
                       start_date_query_key: str, end_date_query_key: str, next_page_func: Callable) -> None:
    """
    Process all the API normal where this api has not dependencies and can be retrieved directly from the api.
    :param api: Api being processed
    :param access_token: Access_token for authentication to the api
    :param base_uri: Base url of the api. For example, "https://api.degreed.com/api/"
    :param all_dates: Some apis require to use timestamp in the request.
    Degreed api allow date range not more than 7 days. all_dates is a list of date ranges
    [(x_1,y_1),(x_2,y_2),...,(x_n,y_n)] where y-x <= 7 days.
    :return:
    """
    name = api["name"]

    logger.info("Processing '{}'".format(name))

    data = []

    if "needTimeStamp" in api and api["needTimeStamp"] is True:
        data = await _process_api_with_timestamp(api, access_token, base_uri, all_dates, date_format,
                                                 start_date_query_key, end_date_query_key, next_page_func)
    else:
        data = await _process_api_without_timestamp(api, access_token, base_uri, next_page_func)

    data = data_normalizer.transform_dataset(api, data)

    data = data_aggregator.aggregate_dataset(api, data)

    file_path = output_file_path(api, app_run_time, target_data_path)

    logger.info("Downloaded files: '{}'".format(file_path))

    record_count = len(data)

    logger.info("Df length: '{}'".format(record_count))

    if record_count > 0:
        _write_to_file(os.path.abspath(file_path), data)


def _is_int(s) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _all_values(json_dict: Dict, path: list) -> List[str]:
    """
    Returns a list of elements mentioned in the last element of the supplied path in json.
    :param json_dict: json object
    :param path:
    json_dict = {
    'a': 5,
    'b' : {
        'c' : [
            {
                'f' : [
                    {'d' : "456"},
                    {'d' : "123"}
                ]
            },
            {
                'f' : [
                         {'d' : "43456"},
                         {'d' : "12343"}
                     ]
            }
        ]
     }
    }
    path = ["a"], OUTPUT = [5]
    path = ["b","c", "f", "d"], OUTPUT = ['456', '123', '43456', '12343']
    path = ["b","c","1","f","d"], OUTPUT = ['43456', '12343']
    :return: list of values extracted from the json
    """
    collected = []

    def get_specific_element_next_iter_function(index: int, items: list) -> Callable[[List[str]], None]:
        """
        For a given index, select the element from the items of node and return the function for the next iteration.
        If index does not exist, return a no-op function. If the index exists, return the function with the element
        selected in the next loop so that rest of the path can be visited.
        :param index: index of the element to be selected
        :param items: collection of items to be selected from
        :return:
        """

        # though r isn't used, it is needed to satisfy return signature of the function
        def no_op(rest: list):
            return

        def next_loop(rest: list):
            return iterate_json(items[index], rest)

        if index < 0:
            raise RuntimeError("Negative index error: {}".format(index))

        if index >= len(items):
            return no_op

        return next_loop

    def iterate_json(node, remaining_path: list) -> None:
        """
        Recursively goes through all the element specified in the remaining_path and
        adds the value retrieved by the last key in the list.
        :param node:
        :param remaining_path:
        :return:
        """

        if not remaining_path:
            return

        # get the first key from the list
        key, *tail = remaining_path

        # get the value
        value = node[key]

        if type(value) not in [Dict, list] and tail:
            raise RuntimeError("Value {} cannot be retrieved from value '{}'".format(tail, value))

        # if the type of the value is another dict, loop again with the tail
        if type(value) is dict:
            iterate_json(value, tail)
        elif type(value) is list:
            # if the type of the value is list, loop again for each element or
            # loop just another time with a specific
            # element in the list
            tail_head, *rest = tail

            if _is_int(tail_head):
                func = get_specific_element_next_iter_function(int(tail_head), value)
                return func(rest)

            for v in value:
                iterate_json(v, tail)
        else:
            # the value is of primitive type, final value reached
            collected.append(value)

    iterate_json(json_dict, path)

    return collected


async def _process_depended_api(api, parent_api, access_token, base_uri, app_run_time, target_data_path,
                                next_page_func: Callable) -> None:
    """
    Process the api without any time restriction
    :param api: Api being processed
    :param access_token: Access_token for authentication to the api
    :param base_uri: Base url of the api. For example, "https://api.degreed.com/api/"
    """
    name = api["name"]

    logger.info("Processing '{}'".format(name))

    api_path = api["path"]

    parent_data_file_path = Path(output_file_path(parent_api, app_run_time, target_data_path))

    collected = []

    extract_func = _get_extract_function(api)

    filter_funcs = filter_element.get_filter_function(api)

    async def fetch(path: str) -> Union[None, List]:
        url = base_uri + path
        result = await fetch_json_to_df(url, access_token, filter_funcs, extract_func, next_page_func)
        return result

    async def process_query(source_data: Dict) -> None:
        """
        This function performs the traversal logic defined by the api on the source_data json record.
        The traversal can yield multiple rows. For each row, replace the replacementParam part of the path in api
        with row value. Make API call on this new path to retrieve data and put it into collected result.
        :param source_data: JSON record
        :return:
        """
        queries = api["query"]

        for query in queries:

            json_path_list = [x for x in query["replacementBy"].split(".") if x.strip()]

            values = _all_values(source_data, json_path_list)

            for v in values:
                new_path = api_path.replace(query["replacementParam"], v)
                res = await fetch(new_path)
                collected.extend(res)

    if parent_data_file_path.exists():  # before opening the file, make sure it exists
        with parent_data_file_path.open("r") as f:

            line = f.readline()

            while line:
                data = json.loads(line)

                await process_query(data)

                line = f.readline()

            f.close()

    file_path = output_file_path(api, app_run_time, target_data_path)

    logger.info("Downloaded files: '{}'".format(file_path))

    if collected:
        _write_to_file(os.path.abspath(file_path), collected)


async def _process_api_with_parent(api, parents, access_token, base_uri, app_run_time, target_data_path,
                                   next_page_func: Callable):
    """
    Function to kick off ingestion of the supplied api that has a parent api requirement
    :param api: Api being processed
    :param parents: This is actually a list.
    It is important that there is only 1 element in the list as there cannot be more than two parents...today.
    :param access_token: Access_token for authentication to the api
    :param base_uri: Base url of the api. For example, "https://api.degreed.com/api/"
    :param app_run_time: Time of the application run
    """
    if parents and len(parents) == 1:
        parent = parents[0]
        await _process_depended_api(api, parent, access_token, base_uri, app_run_time, target_data_path, next_page_func)
    elif len(parents) > 1:
        logger.error("Duplicate parents. Skipping '{}' ...".format(api["name"]))
        raise RuntimeError("There are multiple api with the same name of {}".format(api["parent"]))
    else:
        raise RuntimeError(("No parent data '{}'. Skipping '{}' ...".format(api["parent"], api["name"])))


def _get_next_page_url(path: List[str], response: Dict) -> Union[None, str]:
    """
    Gets the url from the json response to get next page data using the path
    :param path:
    :param response:
    :return:
    """

    def get_next(path_list: List[str], json_dict: Dict):

        # if the path is empty, there is nothing to retrieve and not more data to retrieve from json response
        if not path_list:
            return None

        # get the first key from the list
        key, *tail = path_list

        # if the json response does not have the key, notify the user. No next page data
        if key not in json_dict:
            return None

        if json_dict[key] is None:
            logger.warning("Key is present but the value is none. Key = {}".format(key))
            return None
        elif not tail:
            # if the tail is empty, then we have reached out final node in the json. Just return the value.
            # this is the next url for next set of data
            return json_dict[key]
        else:
            # key is present and there is still more keys to traverse.
            # keep traversing
            return get_next(tail, json_dict[key])

    return get_next(path, response)


async def _process(config: Dict,
                   api: Dict,
                   access_token: str,
                   base_uri: str,
                   all_dates: List[Tuple[datetime, datetime]],
                   app_run_time: str,
                   target_data_path: str,
                   start_date_query_key: str,
                   end_date_query_key: str,
                   date_format: str):
    """
    Function to kick off ingestion of the supplied api
    :param config: application config
    :param api: Api being processed
    :param access_token: Access_token for authentication to the api
    :param base_uri: Base url of the api. For example, "https://api.degreed.com/api/"
    :param all_dates: Some apis require to use timestamp in the request.
    Degreed api allow date range not more than 7 days. all_dates is a list of date ranges
    [(x_1,y_1),(x_2,y_2),...,(x_n,y_n)] where y-x <= 7 days.
    :param app_run_time: Time of the application run
    """
    next_page_url_path = []

    if "nextPageUrlPath" in config:
        next_page_url_path = [x for x in config["nextPageUrlPath"].split(".") if x.strip()]

    next_page_func = partial(_get_next_page_url, next_page_url_path)

    if "parent" not in api:
        await _process_api(api, access_token, base_uri, all_dates, app_run_time, target_data_path, date_format,
                           start_date_query_key, end_date_query_key,
                           next_page_func)
    else:
        parents = [x for x in config["apis"] if x["name"] == api["parent"]]
        await _process_api_with_parent(api, parents, access_token, base_uri, app_run_time, target_data_path,
                                       next_page_func)


def _get_history_metadata(args: Dict) -> Dict:
    """
    TODO We need to get the dates from an external source so that we know till what time we have ingested data.
    :param args:
    :return:
    """

    s3 = boto3.resource("s3")

    obj = s3.Object(args["s3_bucket"], args["metadata_file_key"])

    try:
        body = obj.get()['Body'].read()

        json_dict = json.loads(body)

        return json_dict
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            logger.info('No object found - returning empty metadata')

            empty_dict = {
                "apis": []
            }

            return empty_dict
        else:
            raise


def _get_next_ingestion_dates(metadata_dict: Dict,
                              default_start_date: datetime,
                              default_end_date: datetime,
                              day_range: str,
                              date_format: str,
                              api: dict) -> List[Tuple[date, date]]:
    values = metadata_dict["apis"]

    api_meta = [x for x in values if x["name"] == api["name"]]

    if len(api_meta) > 1:  # cannot have more than one meta data items for the same api
        raise RuntimeError("Multiple metadata values for: {}".format(api["name"]))
    elif api_meta:
        # at this time there is only one api in the list.
        # get the last date, add the day_range, and return a list of one element
        meta = api_meta[0]

        # NOTE: timedelta(seconds=1) is very specific to NY Reg 187.
        # Today api gives data inclusive of both start and end date times.
        s_date = datetime.strptime(meta["last_date"], date_format)

        if s_date.tzinfo is None:
            start_date = timezone.localize(s_date)
        else:
            start_date = s_date

        end_date = timezone.localize(datetime.now())

        if start_date >= end_date:  # should not really happen..
            return []

        return get_start_end_dates(start_date, end_date, day_range)
        # return [(start_date, end_date)]
    else:
        # if no information is available for the given api, resort to default start and end date
        return get_start_end_dates(default_start_date, default_end_date, day_range)


def _add_to_metadata(date_format: str, current_meta: Dict, api_name: str, last_date_time: date) -> Dict:
    """
    Returns updated metadata where the api_name is updated or added with new last_date information
    :param date_format: datetime format
    :param current_meta: current state of metadata json
    :param api_name: name of the api being handled
    :param last_date_time: new laste date time value
    :return:
    """
    current_api_metadata = current_meta["apis"].copy()
    current_api_meta = [x for x in current_api_metadata if x["name"] == api_name]

    def update_list_with_new(old: Dict, new: Dict):
        if old["name"] == api_name:
            return new
        return old

    if len(current_api_meta) > 1:
        raise RuntimeError("Multiple metadata values for: {}".format(api_name))
    elif current_api_meta:
        api_new_meta = current_api_meta[0]

        api_new_meta.update(last_date=last_date_time.strftime(date_format))

        new_list = list(map(lambda x: update_list_with_new(x, api_new_meta), current_api_metadata))

        current_meta.update(apis=new_list)

        return current_meta
    else:
        new_api_meta = {
            "name": api_name,
            "last_date": last_date_time.strftime(date_format)
        }

        curr_list = current_meta["apis"].copy()

        curr_list.append(new_api_meta)

        current_meta.update(apis=curr_list)

        return current_meta


def _write_metadata(metadata: Dict, s3_bucket: str, path: str):
    s3 = boto3.resource("s3")
    s3.Object(s3_bucket, path).put(Body=json.dumps(metadata))


async def main():
    args = parse_args()

    config_file = args.conf_file

    with open(config_file, "r") as config_fh:
        config = yaml.safe_load(config_fh)

    base_uri = config["baseUri"]

    token_uri = config["tokenUri"]

    target_data_path = config["targetPath"]

    app_run_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

    total_time = 0

    independent_apis = [x for x in config["apis"] if "parent" not in x]

    dependent_apis = [x for x in config["apis"] if "parent" in x]

    # get the delta day range, default start and end dates
    delta_day_range = args.delta_day_range

    default_start_date = datetime.strptime(args.default_start_date, args.meta_date_format)

    default_end_date = datetime.strptime(args.default_end_date, args.meta_date_format)

    # load the history meta data from s3
    history_meta = _get_history_metadata(vars(args))

    # we are going to generate a new metadata. Initialize it existing state
    metadata = history_meta.copy()

    # partial function that will enable call to update the metadata json dict by supplying current metadata. api
    # name, and last_date
    metadata_updater = partial(_add_to_metadata, args.meta_date_format)

    # partial function that will enable to get next ingestion dates for a given api
    get_dates = partial(_get_next_ingestion_dates, history_meta,
                        default_start_date, default_end_date,
                        delta_day_range, args.meta_date_format)

    start_date_query_key = ""
    if "startDateQueryKey" in config:
        start_date_query_key = config["startDateQueryKey"]

    end_date_query_key = ""
    if "endDateQueryKey" in config:
        end_date_query_key = config["endDateQueryKey"]

    # we need to process independent apis before the ones that are depended ones.
    # depended apis are depended on independent apis. depended apis read the data downloaded by independent api.
    # typical use case is where we get ALL PATHWAYS (independent api).
    # Then foreach of the PATHWAY with id, we call another API
    # to get more information about this PATHWAY like CONTENT it refers
    for api in independent_apis + dependent_apis:
        logger.info("Processing {} api".format(api['name']))
        dates = get_dates(api)

        start_time = time.time()
        await _process(config, api,args.access_token, base_uri, dates, app_run_time, target_data_path,
                       start_date_query_key, end_date_query_key, args.meta_date_format)
        elapsed_time = time.time() - start_time

        last_date = dates[-1][1]

        metadata = metadata_updater(metadata, api["name"], last_date)

        logger.info("Time taken to process '{}': '{}'".format(api["name"], elapsed_time))

        total_time = total_time + elapsed_time

    logger.info("Total Time taken: '{}'".format(total_time))

    _write_metadata(metadata, args.s3_bucket, args.metadata_file_key)

    logger.info("Updated historical metadata...")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
