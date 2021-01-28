ALTER SESSION SET FlexTableRawSize = 10000000;

-- First create the flex table:
DROP TABLE IF EXISTS :schema_name.linkedin_analytics_stg;

CREATE FLEX TABLE :schema_name.linkedin_analytics_stg();
COPY :schema_name.linkedin_analytics_stg FROM :src_file
PARSER fjsonparser(start_point='elements');


