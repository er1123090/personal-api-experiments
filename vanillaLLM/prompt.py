
IMPLICIT_ZS_PROMPT_TEMPLATE = """You are an intelligent API selection assistant.
Your goal is to generate the correct API call based on the user's current request and their dialogue history.

[Reasoning Steps]
1. **Analyze History**: Look for consistent patterns/preferences across past sessions. These are the user's preferences.
2. **Analyze Current Request**: Identify what the user requests right now.
3. **Resolve Conflicts**: 
   - If the current request specifies a value, USE IT (even if it contradicts history).
   - If the current request is silent on a slot, FILL IT using the historical preference.
   - Do not invent values that are neither in history nor in the current request.

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}


<API_CALL>
function_name(arg1="value", arg2="value", ...)
"""

IMPLICIT_FS_PROMPT_TEMPLATE = """You are an intelligent API selection assistant.
Your goal is to generate the correct API call based on the user's current request and their dialogue history.

[Reasoning Steps]
1. **Analyze History**: Look for consistent patterns/preferences across past sessions. These are the user's preferences.
2. **Analyze Current Request**: Identify what the user requests right now.
3. **Resolve Conflicts**: 
   - If the current request specifies a value, USE IT (even if it contradicts history).
   - If the current request is silent on a slot, FILL IT using the historical preference.
   - Do not invent values that are neither in history nor in the current request.

Here are examples of how to infer implicit preferences:

[Example 1: Cross-Domain Preference Transfer]
Dialogue History:
User: "Book a cheap hotel for tonight." -> GetHotels(price_range="cheap", ...)
User: "Call a taxi for me." -> GetRideSharing(ride_type="shared", ...)
User Utterance:
"I need a restaurant reservation nearby."
Output:
<API_CALL>
GetRestaurants(price_range="cheap", location="nearby")

[Example 2: Consistency in Group Size]
Dialogue History:
User: "One ticket for the movie." -> GetEvents(tickets=1, ...)
User: "Book a single room." -> GetHotels(room_type="single", ...)
User Utterance:
"Look for a flight to New York."
Output:
<API_CALL>
GetFlights(destination="New York", passengers=1)

[Example 3: Explicit Override of Implicit Preference]
Dialogue History:
User: "Find a budget hostel." -> GetHotels(price_range="cheap", ...)
User: "I need the cheapest flight." -> GetFlights(flight_class="Economy", ...)
User Utterance:
"I want to celebrate today, find me a fancy luxury restaurant."
Output:
<API_CALL>
GetRestaurants(price_range="expensive", sort_by="rating")

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""



IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE = """You are an intelligent API selection assistant.
Your goal is to generate the correct API call based on the user's current request and their dialogue history.

[Reasoning Steps]
1. **Analyze History**: Look for consistent patterns/preferences across past sessions. These are the user's preferences.
2. **Analyze Current Request**: Identify what the user requests right now.
3. **Resolve Conflicts**: 
   - If the current request specifies a value, USE IT (even if it contradicts history).
   - If the current request is silent on a slot, FILL IT using the historical preference.
   - Do not invent values that are neither in history nor in the current request.

   
[Preference Groups]:
{{
  "low_cost": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetRestaurants", "slot": "price_range", "value": "cheap" }},
      {{ "domain": "GetRentalCars", "slot": "car_type", "value": "Compact" }},
      {{ "domain": "GetHotels", "slot": "average_star", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "average_star", "value": 2 }},
      {{ "domain": "GetRideSharing", "slot": "shared_ride", "value": true }},
      {{ "domain": "GetHotels", "slot": "free_entry", "value": true }},
      {{ "domain": "GetFlights", "slot": "flight_class", "value": "Economy" }}
    ]
  }},
  "high_cost": {{
    "group_preference": "luxury_preference",
    "rules": [
      {{ "domain": "GetRestaurants", "slot": "price_range", "value": "pricey" }},
      {{ "domain": "GetRentalCars", "slot": "car_type", "value": "Full-size" }},
      {{ "domain": "GetHotels", "slot": "average_star", "value": 4 }},
      {{ "domain": "GetHotels", "slot": "average_star", "value": 5 }}
    ]
  }},
  "solo_usage": {{
    "group_preference": "solo_travel",
    "rules": [
      {{ "domain": "GetBuses", "slot": "group_size", "value": 1 }},
      {{ "domain": "GetFlights", "slot": "passengers", "value": 1 }},
      {{ "domain": "GetRideSharing", "slot": "number_of_seats", "value": 1 }},
      {{ "domain": "GetEvents", "slot": "number_of_tickets", "value": 1 }}
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