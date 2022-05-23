select this_week, previous_week, -uniq(user_id) as num_users, status
from
(
  select user_id, 
  groupUniqArray(toMonday(toDate(time))) as weeks_visited,
  --Collecting all Mondays of all weeks user had an action entry in (including this week)
  addWeeks(arrayJoin(weeks_visited), +1) this_week,
  --Getting next Monday (next week) for all weeks_visited and calling it... this_week: because this we take a step back and thus weeks_visited are previous for us!
  if(has(weeks_visited, this_week) = 1, 'retained', 'gone') as status,
  --Assigning status: user who had action entries both this week and previous is 'retained', user who had not this week is 'gone'
  arrayJoin(weeks_visited) as previous_week
  -- addWeeks(this_week, -1) as previous_week
  --Getting Monday previous to next -- effectively collapsing weeks_visited as if we used 'arrayJoin(weeks_visited)', results are the same, so I'll do just that

  from simulator_20220320.feed_actions
  group by user_id
)

where status = 'gone'
--Gathering only 'gone' users, hence the minus in num_users
group by this_week, previous_week, status

having this_week != addWeeks(toMonday(today()), +1)
--Not including next week (since it has not been fully filled yet -- we would have got it in this_week since we add 1 week)

union all

select this_week, previous_week, toInt64(uniq(user_id)) as num_users, status
from
(
  select user_id, 
  groupUniqArray(toMonday(toDate(time))) as weeks_visited,
  arrayJoin(weeks_visited) this_week, 
  addWeeks(this_week, -1) as previous_week,
  if(has(weeks_visited, previous_week) = 1, 'retained', 'new') as status
  --Assigning status: user who had action entries both this week and previous is 'retained', user who had not this week is 'gone'

  from simulator_20220320.feed_actions
  group by user_id
)

group by this_week, previous_week, status