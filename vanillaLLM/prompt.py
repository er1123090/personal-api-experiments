EXPLICIT_ZS_PROMPT_TEMPLATE = """
You are an API selection assistant. 
Given the user's dialogue and the user's standing instructions (user profile), generate the correct API call. 

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""




EXPLICIT_FS_PROMPT_TEMPLATE = """You are an API selection assistant. 
Given the user's dialogue and the User Utterance, generate the correct API call. 

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)



Example :
    Dialogue History: 
[Session 1]
User: I'm thinking about grabbing dinner later. Any ideas on what I should consider?
Assistant: Sure! Are you leaning toward any particular type of dining or just exploring options?
User: Mostly exploring. I haven't decided what I want yet.
Assistant: Totally fine. There are plenty of places around. Are you thinking casual or something nicer?
User: Not sure. Maybe something moderately nice.
Assistant: That works. Do you usually look for something specific when choosing a restaurant?
User: Well, when I request restaurants, my go-to cuisine is Indian.
Assistant: Got it. In that case, there are several Indian restaurants nearby that have great reviews.
User: That sounds good. Do you know if any of them are usually crowded?
Assistant: Some can be during peak times, but it varies. Weekdays tend to be less busy.
User: Good to know. What's the typical wait time on weekends?
Assistant: Usually around 20 to 30 minutes, depending on the location.
User: Makes sense. Changing the topic—did you have a good day today?
Assistant: Pretty average overall. How about yours?
User: Same here, nothing too dramatic.
Assistant: Those days can be nice sometimes.
User: Yeah, gives me time to relax a bit.
Assistant: Relaxing evenings can definitely help reset the mind.
User: Thanks for the help today.
Assistant: Anytime! Let me know if you need more ideas.

[Session 2]
User: I'm thinking about going out to eat later. Do you have any suggestions?
Assistant: Sure! What kind of dining experience are you in the mood for?
User: Not sure yet. Maybe something a bit nicer.
Assistant: Great! There are several upscale options depending on the cuisine you prefer.
User: I haven’t decided on cuisine.
Assistant: That's okay. Are you celebrating something or just eating out casually?
User: Actually, when I’m requesting restaurants, I prefer the Indian restaurant named Amber with an expensive price range.
Assistant: Got it. That restaurant is known for its high-end Indian dishes and elegant atmosphere.
User: Sounds great. Do you know if reservations are needed?
Assistant: It’s usually recommended, especially during dinner hours.
User: Okay, I’ll keep that in mind. Do they have good reviews?
Assistant: Yes, they consistently receive positive feedback for both food and service.
User: Nice. Changing the topic—did you watch anything interesting recently?
Assistant: Not really, just some light shows in the background.
User: Same here. Nothing new to watch lately.
Assistant: It happens. Some weeks are just quieter for entertainment.
User: True. Maybe I’ll find something this weekend.
Assistant: Hope you find something enjoyable!
User: Thanks for your help.
Assistant: You're welcome! Happy to assist.

[Session 3]
User: I’ve been thinking about doing a short getaway soon. Not sure where yet.
Assistant: That sounds refreshing. Are you leaning toward nature, city, or something quiet?
User: Not sure. Maybe just somewhere relaxing.
Assistant: There are many calm places depending on how far you'd like to travel.
User: True, I should narrow it down somehow.
Assistant: Yes, picking an accommodation type could help—hotel, houses, or rentals.
User: If I’m requesting HouseStays, I prefer Portland, OR as my destination.
Assistant: Got it. Portland has many cozy house-style stays in peaceful neighborhoods.
User: That sounds nice. Are most of them near downtown?
Assistant: Quite a few are within a short drive, yes.
User: Cool. What’s the weather usually like there?
Assistant: A bit mild, sometimes rainy, but very manageable.
User: Good to know. On another note, how’s your week going?
Assistant: Pretty normal so far. Nothing unusual.
User: Mine too. Just trying to stay productive.
Assistant: That’s always a good goal.
User: Maybe I’ll plan something this weekend.
Assistant: Hope it turns out relaxing for you!
User: Thanks, I’ll let you know.
Assistant: Anytime!

[Session 4]
User: I might plan a trip soon but haven’t decided on the details yet.
Assistant: Planning ahead can make things smoother. What kind of trip are you imagining?
User: Not sure, maybe something calm.
Assistant: That sounds nice. Do you already know who you're traveling with?
User: Yes, I have a general idea.
Assistant: Great! Lodging and activities vary a lot depending on the group.
User: Whenever I plan travel, I request that my travel be suitable for children.
Assistant: Understood. In that case, family-friendly options are especially important.
User: Exactly. Do those places usually have activities?
Assistant: Most do—like small play areas or relaxed walking paths.
User: Good to know. Does it tend to be expensive?
Assistant: It depends on the destination, but many budget-friendly choices exist.
User: Makes sense. Changing the topic—how's the weather lately?
Assistant: Pretty mild. Some days are sunny, some are cloudy.
User: Ah, typical week then.
Assistant: Seems like it.
User: I’ll think more about trip ideas later.
Assistant: Let me know if you want suggestions!
User: Thanks.
Assistant: Anytime!

[Session 5]
User: I’m thinking of going out this weekend. Maybe to some event.
Assistant: Events can be fun. Are you into performances, festivals, or something casual?
User: Not sure yet, still browsing.
Assistant: There’s a lot happening depending on the city.
User: Yeah, I should narrow things down.
Assistant: True. Some people choose by category—music, art, food, etc.
User: I’m looking for an event with the event type Music, category Pop, and the name Aly And Aj.
Assistant: Got it. That's a popular pop act and tends to draw a lively crowd.
User: Sounds interesting. Are tickets usually hard to get?
Assistant: They can sell out but usually not instantly.
User: Good to know. What about parking?
Assistant: Varies by venue, but many places offer nearby lots.
User: Alright. On another note, did anything fun happen today?
Assistant: Nothing too exciting, just a typical day.
User: Same here.
Assistant: Some days are just simple like that.
User: Maybe I’ll check events later tonight.
Assistant: Hope you find something enjoyable.
User: Thanks for the info.
Assistant: You're welcome!

[Session 6]
User: I might take a short trip soon but haven’t decided where.
Assistant: That sounds relaxing. City or countryside?
User: Not sure, maybe something calm.
Assistant: A peaceful spot could be a great break.
User: Yeah, I could use that.
Assistant: Do you already know what kind of stay you want?
User: When searching for HouseStays, I look for ones in Portland, OR with a 4.5 rating and laundry service.
Assistant: Got it. Those criteria usually point to clean, well-managed places.
User: Do they tend to be pricey?
Assistant: Depends on the season, but many mid-range options exist.
User: Good to know. What about transportation?
Assistant: Portland has good public transit and walkable areas.
User: Sounds convenient. How’s your day going?
Assistant: Pretty standard so far.
User: Mine too. Just the usual.
Assistant: That’s how it goes sometimes.
User: Maybe I’ll look at travel sites later.
Assistant: Hope you find something nice!
User: Thanks.
Assistant: Anytime!

[Session 7]
User: I’m thinking of getting a trim sometime soon.
Assistant: Sounds good. Are you thinking of going to a salon or barbershop?
User: Not sure yet. Haven’t chosen a place.
Assistant: There are plenty of places depending on the style you're looking for.
User: Yeah, I need to narrow it down.
Assistant: Do you prefer a particular stylist or service quality?
User: When I request salon services, I ask for A Businessman's Haircut as my stylist.
Assistant: Understood. That’s a popular place known for its clean cuts.
User: Do they take walk-ins?
Assistant: Usually yes, but appointments are more reliable.
User: Got it. How long is the wait usually?
Assistant: Often around 10 to 20 minutes.
User: Alright. Changing topic—how’s your day?
Assistant: Pretty calm overall.
User: Mine too. Nothing special.
Assistant: Some days are just ordinary.
User: Maybe I’ll visit later this week.
Assistant: Hope it turns out well!
User: Thanks.
Assistant: You're welcome!

[Session 8]
User: I'm thinking about watching a movie tonight.
Assistant: Nice! Are you in the mood for something specific?
User: Not sure, maybe something light.
Assistant: There are lots of options depending on the genre you feel like.
User: Yeah, still deciding.
Assistant: Evening showtimes can be flexible depending on the theatre.
User: Whenever I request movies, I ask for showtimes at 10:30 pm.
Assistant: Got it. Late-night showings usually have smaller crowds too.
User: That’s good. Are seats usually available?
Assistant: Typically yes, especially on weekdays.
User: Cool. What about snacks?
Assistant: Most theatres have standard popcorn, drinks, and candy.
User: Nice. By the way, how’s your day?
Assistant: Pretty uneventful, honestly.
User: Mine too. Nothing new.
Assistant: Some days are just like that.
User: Maybe I’ll check a listing later.
Assistant: Sounds like a plan!
User: Alright, thanks!
Assistant: Happy to help!

User Utterance:
User: I want to eat at a costly place.
Agent: What city?
User: SF.


API_CALL :
    GetRestaurants(city="SF", price_range="expensive", cuisine="Indian", restaurant_name="Amber")
"""




IMPLICIT_ZS_PROMPT_TEMPLATE = """You are an API selection assistant.
Given the user's dialogue history and the current user utterance, generate the correct API call.

Important:
- Infer the user's **implicit preferences** ONLY from repeated behavior across past sessions and/or repeated slot-value patterns in past API calls.
- Do NOT treat one-off requests as preferences unless they recur.
- If the current utterance explicitly overrides an inferred preference, follow the current utterance.
- Only use domains and slots allowed by the provided schema. Do not invent new ones.
- Output ONLY one API call.

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""

IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE = """You are an API selection assistant.
Given the user's dialogue history and the current user utterance, generate the correct API call.

Important:
- Infer the user's **implicit preferences** ONLY from repeated behavior across past sessions and/or repeated slot-value patterns in past API calls.
- domain-slot-values in the same preference group are considered equivalent.
- Do NOT treat one-off requests as preferences unless they recur.
- If the current utterance explicitly overrides an inferred preference, follow the current utterance.
- Only use domains and slots allowed by the provided schema. Do not invent new ones.
- Output ONLY one API call.

Preference Groups:
{{
  "low_cost": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetRestaurants", "slot": "price_range", "value": "cheap" }},
      {{ "domain": "GetFlights", "slot": "flight_class", "value": "economy" }},
      {{ "domain": "GetFlights", "slot": "flight_class", "value": "coach" }},
      {{ "domain": "GetHotels", "slot": "average_rating", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "average_rating", "value": 2 }},
      {{ "domain": "GetRentalCars", "slot": "car_type", "value": "Compact" }}
    ]
  }},

  "mid_cost": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetRestaurants", "slot": "price_range", "value": "moderate" }},
      {{ "domain": "GetHotels", "slot": "average_rating", "value": 3 }},
      {{ "domain": "GetRentalCars", "slot": "car_type", "value": "Standard" }}
    ]
  }},

  "cost_avoidance": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetTravel", "slot": "free_entry", "value": true }},
      {{ "domain": "GetRideSharing", "slot": "shared_ride", "value": true }}
    ]
  }},

  "minimal_usage": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetHomes", "slot": "number_of_beds", "value": 1 }},
      {{ "domain": "GetHomes", "slot": "number_of_baths", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "number_of_rooms", "value": 1 }},
      {{ "domain": "GetEvents", "slot": "number_of_tickets", "value": 1 }}
    ]
  }},

  "solo_usage": {{
    "group_preference": "solo_travel",
    "rules": [
      {{ "domain": "GetFlights", "slot": "passengers", "value": 1 }},
      {{ "domain": "GetBuses", "slot": "group_size", "value": 1 }},
      {{ "domain": "GetRideSharing", "slot": "number_of_seats", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "number_of_rooms", "value": 1 }},
      {{ "domain": "GetHomes", "slot": "number_of_beds", "value": 1 }}
    ]
  }},

  "low_friction": {{
    "group_preference": "low_friction",
    "rules": [
      {{ "domain": "GetHotels", "slot": "has_wifi", "value": true }},
      {{ "domain": "GetTravel", "slot": "free_entry", "value": true }},
      {{ "domain": "GetRentalCars", "slot": "pickup_location", "value": "airport" }},
      {{ "domain": "GetRideSharing", "slot": "shared_ride", "value": true }}
    ]
  }}
}}


Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""


IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE = """You are an API selection assistant.
Given the retrieved long-term memories, the user's dialogue history, and the current user utterance, generate the correct API call.

Important:
- Infer the user's **implicit preferences** ONLY from repeated behavior across past sessions and/or repeated slot-value patterns in past API calls.
- Do NOT treat one-off requests as preferences unless they recur.
- If the current utterance explicitly overrides an inferred preference, follow the current utterance.
- Only use domains and slots allowed by the provided schema. Do not invent new ones.
- Output ONLY one API call.


Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)

Now produce the correct API call:
"""