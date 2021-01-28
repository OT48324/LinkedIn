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
        glob_pattern='linkedin_analytics-????-??-??-??-??-??.json',
        destination_dir='linkedin_analytics',
        table_name='linkedin_analytics_stg',
        schema_script='sql/linkedin/analytics/linkedin_analytics_schema.sql',
        load_script='sql/linkedin/analytics/linkedin_analytics_load.sql',
    ),
     DataSpec(
        glob_pattern='linkedin_campaign-????-??-??-??-??-??.json',
        destination_dir='linkedin_campaign',
        table_name='linkedin_campaign_stg',
        schema_script='sql/linkedin/campaign/linkedin_campaign_schema.sql',
        load_script='sql/linkedin/campaign/linkedin_campaign_load.sql',
    )
    ]
