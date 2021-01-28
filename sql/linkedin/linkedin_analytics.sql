Insert into edw_staging.linkedin_analytics
select
cast( __raw__ ['actionClicks']	AS VARCHAR ) AS	actionClicks,
cast( __raw__ ['adUnitClicks']	AS VARCHAR ) AS	adUnitClicks,
cast( __raw__ ['clicks']	AS VARCHAR ) AS	clicks,
cast( __raw__ ['costInUsd']	AS VARCHAR ) AS	costInUsd,
cast( __raw__ ['dateRange.start.day']	AS VARCHAR ) AS	day,
cast( __raw__ ['dateRange.start.month']	AS VARCHAR ) AS	month,
cast( __raw__ ['dateRange.start.year']	AS VARCHAR ) AS	year,
cast( __raw__ ['follows']	AS VARCHAR ) AS	follows,
cast( __raw__ ['impressions']	AS VARCHAR ) AS	impressions,
cast( __raw__ ['landingPageClicks']	AS VARCHAR ) AS	landingPageClicks,
cast( __raw__ ['leadGenerationMailInterestedClicks']	AS VARCHAR ) AS	leadGenerationMailInterestedClicks,
cast( __raw__ ['likes']	AS VARCHAR ) AS	likes,
cast( __raw__ ['oneClickLeadFormOpens']	AS VARCHAR ) AS	oneClickLeadFormOpens,
cast( __raw__ ['oneClickLeads']	AS VARCHAR ) AS	oneClickLeads,
cast( __raw__ ['opens']	AS VARCHAR ) AS	opens,
cast( __raw__ ['otherEngagements']	AS VARCHAR ) AS	otherEngagements,
cast( __raw__ ['pivotValues']['0']	AS VARCHAR ) AS	campaign_ID,
cast( __raw__ ['sends']	AS VARCHAR ) AS	sends,
cast( __raw__ ['shares']	AS VARCHAR ) AS	shares,
cast( __raw__ ['totalEngagements']	AS VARCHAR ) AS	totalEngagements
FROM edw_staging.linkedin_analytics_stg;
