Insert into edw_staging.linkedin_campaign 
SELECT cast( __raw__[ 'account' ] AS VARCHAR ) AS account,
cast( __raw__[ 'campaignGroup' ] AS VARCHAR ) AS campaignGroup, 
cast( __raw__[ 'creativeSelection' ] AS VARCHAR ) AS creativeSelection, 
cast( __raw__[ 'dailyBudget.amount' ] AS VARCHAR ) AS dailyBudget, 
cast( __raw__[ 'id' ] AS VARCHAR ) AS id, 
cast( __raw__[ 'name' ] AS VARCHAR ) AS name, 
cast( __raw__[ 'servingStatuses' ]['0'] AS VARCHAR ) AS servingStatuses, 
cast( __raw__[ 'totalBudget.amount' ] AS VARCHAR ) AS totalBudget, 
cast( __raw__[ 'type' ] AS VARCHAR ) AS type, 
cast( __raw__[ 'status' ] AS VARCHAR ) AS status 
FROM edw_staging.linkedin_campaign_stg where cast( __raw__[ 'id' ] AS VARCHAR ) NOT in 
(select id from edw_staging.linkedin_campaign) ;
commit;
