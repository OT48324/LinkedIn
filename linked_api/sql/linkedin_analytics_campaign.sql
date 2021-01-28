Truncate table edw_staging.linkedin_analytics_campaign;

Insert into edw_staging.linkedin_analytics_campaign
select 
to_date(concat(concat(concat(CONCAT("day",'/'),"month"),'/'),"year"),'dd/mm/yyyy')as record_date,
SUBSTRING(account,25) as account_id,
SUBSTRING(campaignGroup,31) as campaigngroup,
id as campaign_id, 
name as campaign_name,
actionClicks, 
adUnitClicks, 
clicks, 
costInUsd,
follows, 
impressions, 
landingPageClicks, 
leadGenerationMailInterestedClicks, 
likes, 
oneClickLeadFormOpens, 
oneClickLeads, 
opens, 
otherEngagements,
sends, 
shares, 
totalEngagements,
creativeSelection as campaigncreativeSelection, 
dailyBudget as campaigndailyBudget, 
servingStatuses as campaignservingStatuses, 
totalBudget as campaigntotalBudget, 
"type" as campaigntype, 
status as campaignstatus, 
run_schedule_start, 
run_schedule_end
from edw_staging.linkedin_analytics a
inner join edw_staging.linkedin_campaign b on SUBSTRING(a.campaign_id,26) =b.id ;

commit;
