"""
Configuration required for linkedin data ingestion process.
Required variables:
    data_specs
"""
from collections import namedtuple
import preprocessors

DataSpec = namedtuple(
    'DataSpec', [
        'glob_pattern',
        'destination_dir',
        'table_name',
        'schema_script',
        'load_script',
        'preprocessor',
        'date_string',
        'extra_trailing_characters',
        'kwargs'
    ]
)

DataSpec.__new__.__defaults__ = (
    None, None, None, None, None, None, '%Y-%m-%d-%H-%M-%S', None, {}
)

data_specs = [
    DataSpec(
        glob_pattern='linkedin-????-??-??-??-??-??.json',
        destination_dir='linkedin',
        table_name='linkedin',
        schema_script='sql/linkedin/linkedin_schema.sql',
        load_script='sql/linkedin/linkedin_load.sql',
    )
    ]
