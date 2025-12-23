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


IMPLICIT_ZS_PROMPT_CONF_MEMORY_TEMPLATE = """You are an API selection assistant.
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


#====================================================================================================================================================================================

# Recursive Memory Update Prompt (English)
RECURSIVE_MEMORY_UPDATE_PROMPT_V1 = """
You are an intelligent memory manager responsible for maintaining a user's personalization profile over time.
Your goal is to update the 'Explicit' and 'Implicit' preferences based on the interaction history.

[INPUT DATA]
1. Previous Memory State (from t-1):
   - Explicit Preferences: {prev_explicit}
   - Implicit Preferences: {prev_implicit}

2. Current Session Data (at t):
   - Dialogue History (H_t): {h_t}
   - API Call History (S_t): {s_t}

[INSTRUCTIONS]
1. Analyze the 'Current Session Data' to identify new user preferences or changes in behavior.
2. EXPLICIT PREFERENCES (E_t):
   - Extract clear, stated preferences from the user (e.g., "I dislike spicy food", "I prefer economy seats").
   - Update the previous explicit preferences. If the user contradicts a past preference, prioritize the new one.
   - Refer to the previous explicit preferences and update them accordingly.
3. IMPLICIT PREFERENCES (I_t):
   - Deduce patterns, habits, or latent preferences from API parameters and dialogue context.
   - Update the previous implicit preferences.
4. CONSOLIDATION:
   - Merge new findings with the previous state to create a comprehensive profile.
   - Summarize concisely but strictly.

[OUTPUT FORMAT]
You must respond ONLY with a valid JSON object. Do not include any other text.
{{
    "explicit_pref": "Updated summary of explicit preferences...",
    "implicit_pref": "Updated summary of implicit preferences..."
}}
"""

RECURSIVE_MEMORY_UPDATE_PROMPT_V2 = """
You are a long-term personalization memory manager.
Your task is to UPDATE the user's memory state through explicit reasoning and justified actions,
not by appending text, but by performing structured memory operations.

────────────────────────
[INPUT]
1. Previous Memory State (t-1)
   - Explicit Preferences (E_{{t-1}}): {prev_explicit}
   - Implicit Preferences (I_{{t-1}}): {prev_implicit}

2. Current Session Evidence (t)
   - Dialogue History (H_t): {h_t}
   - API Call History (S_t): {s_t}

────────────────────────
[STEP 1: OBSERVATION]
Identify ALL candidate preference signals from the current session.
For each signal, specify:
- Type: explicit | implicit
- Evidence: exact utterance or API pattern
- Strength: weak | medium | strong
- Consistency with past memory: consistent | conflicting | new | ambiguous

────────────────────────
[STEP 2: REASONING]
For each candidate signal, reason carefully:
- Is this a stable preference or a situational behavior?
- Does it repeat, intensify, or contradict past preferences?
- Should it update memory permanently or be ignored for now?

You MUST NOT update memory unless justified by evidence.

────────────────────────
[STEP 3: MEMORY ACTION SELECTION]
For EACH justified update, select ONE action:
- add        : introduce a new preference
- reinforce : strengthen an existing preference
- weaken    : reduce confidence in an existing preference
- override  : replace a conflicting preference
- ignore    : do not update memory

Explain WHY each action is chosen.

────────────────────────
[STEP 4: STATE CONSOLIDATION]
Reconstruct the memory state after applying all actions.
The final memory MUST:
- Be concise
- Reflect stability over time
- Remove outdated or overridden preferences
- Avoid duplicative or vague statements

────────────────────────
[OUTPUT FORMAT]
Return ONLY a valid JSON object.
Do NOT include reasoning steps verbatim.
Explicit_pref and Implicit_pref should be natural language sentences.

{{
  "explicit_pref": "Consolidated explicit preferences after justified updates",
  "implicit_pref": "Consolidated implicit preferences after justified updates"
}}
"""

RECURSIVE_MEMORY_UPDATE_PROMPT_V3 = """
You are a "Behavioral Analyst & Memory Consolidator" for an AI agent.
Your goal is to maintain a cumulative user profile by **reverse-engineering the user's decision-making criteria** from their interaction history.

[INPUT DATA]
1. Previous Memory (Base):
   - Explicit: {prev_explicit}
   - Implicit: {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[CONSTRAINT: NATURAL LANGUAGE ONLY]
- **STRICTLY FORBIDDEN**: Do NOT use JSON syntax, dictionaries, or brackets inside the output strings (e.g., NO {{"domain": "value"}}).
- **REQUIRED**: All information must be stored as **descriptive natural language sentences**.
- **GOAL**: The output should read like a user profile summary, not a database dump.

[CORE PHILOSOPHY: OBSERVATION vs. INTENT]
You must distinguish between the specific action and the underlying intent.
- **Observation (Fact)**: The specific parameters selected in the API call. (Domain-Bound)
- **Intent (Preference)**: The abstract optimization criteria that drove that selection. (Domain-Agnostic)

[REASONING PROTOCOL]
Analyze the input using the following three distinct steps to merge new evidence with previous memory.

**STEP 1: CONSTRAINT RECORDING (The 'What')**
- Identify strict constraints or filters the user applied.
- Record these as specific facts to ensure accurate retrieval of past details.
- **Cumulative Rule**: Do not delete past constraints unless explicitly contradicted. Merge new constraints with the existing list.

**STEP 2: ATTRIBUTE ABSTRACTION (The 'Why')**
- Analyze the selected slot-values. **Detach** the value from its specific domain.
- Determine the **Attribute Dimension** (e.g., Cost, Speed, Quality, Risk, Quantity) and the **Optimization Direction** (e.g., Minimizing, Maximizing, Balancing).
- **Inference Logic**: If a user selects a value that represents an extreme on a scale (e.g., the lowest/highest option available) or a specific quality tier, identify this as a **Behavioral Trait**.

**STEP 3: CROSS-DOMAIN FORMULATION (The 'Prediction')**
- Formulate preferences using **General Terms** that apply across *any* service industry.
- **Avoid Domain-Specific Nouns**: Do not use words limited to the current domain.
- **Use Transferable Concepts**: Use terms like "efficiency," "comfort," "privacy," "budget," or "reliability."
- This allows the system to apply a trait observed in Domain A to make a prediction in Domain B.

[UPDATE INSTRUCTIONS]
- **Explicit_pref**: A consolidated list of ALL specific constraints and facts accumulated over time.
- **Implicit_pref**: A summary of inferred **Behavioral Traits** and **Decision Principles** derived from Step 2 and 3. Describe *how* the user makes decisions, not just *what* they chose.

[OUTPUT FORMAT]
Return ONLY a JSON object.
{{
    "explicit_pref": "Cumulative summary of specific constraints and stated preferences.",
    "implicit_pref": "Inferred behavioral traits and domain-agnostic decision criteria."
}}
"""

RECURSIVE_MEMORY_UPDATE_PROMPT_V4 = """
You are a "Memory Consolidator" responsible for maintaining a natural language profile of a user.
Your goal is to convert raw interaction data into a persistent, human-readable narrative.

[INPUT DATA]
1. Previous Memory (Base):
   - Explicit: {prev_explicit}
   - Implicit: {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[Constraints]
- **Avoid Raw JSON Dumping**: Do not simply copy-paste the full JSON object from the API calls.
- **Use Structured Natural Language**: You MAY use bullet points, clear headings, or concise formats to organize information.
- **Goal**: The output should be a high-quality **User Profile Note** that an agent can quickly read to understand the user's history.

[PROCESSING PROTOCOL]
Analyze the input using the following three steps to merge new evidence with previous memory.

**STEP 1: NARRATIVE FACT INTEGRATION (The 'What')**
- Translate specific API parameters (Slot-Value pairs) and strict constraints into **clear, declarative sentences**.
- **Cumulative Rule**: Start with the sentences from `Previous Memory`. Append new facts from the `Current Session`.
- **Organization**: You may group sentences by topic (e.g., "Regarding Travel: ...", "Regarding Events: ..."), but keep them as text.

**STEP 2: ATTRIBUTE ABSTRACTION (The 'Why')**
- Analyze the choices to find the **Optimization Criteria**.
- **Detach** the specific value from the domain and identify the **Attribute Dimension** (e.g., Cost, Comfort, Speed, Risk) and **Direction** (Min/Max/Balance).
- **Inference**: If the user selects an extreme option (lowest/highest) or a specific tier, record this as a trait.

**STEP 3: CROSS-DOMAIN GENERALIZATION (The 'Prediction')**
- Formulate behavioral traits using **Domain-Agnostic Terms**.
- Create sentences that predict how the user would behave in a completely different situation based on the observed attribute preferences.
- *Example logic (do not copy)*: If a user optimized for X in Domain A, state that "User prioritizes X across services."

[OUTPUT FORMAT]
Return ONLY a JSON object. The values must be **text strings**, not objects.
{{
    "explicit_pref": "A structured text summary of all specific constraints accumulated so far. (e.g., 'For flight requests, the user specified... In restaurant bookings, the user preferred...')",
    "implicit_pref": "A descriptive summary of inferred behavioral traits and decision-making styles. (e.g., 'The user consistently prioritizes efficiency and reliability over cost...')"
}}
"""

RECURSIVE_MEMORY_UPDATE_PROMPT_IMPLICIT_ONLY_V4 = """
You are a "Memory Consolidator" responsible for maintaining a psychological and behavioral profile of a user.
Your goal is to extract **implicit decision-making patterns** from interaction data, strictly ignoring specific factual constraints or entity details.

[INPUT DATA]
1. Previous Preference (Base):
   - {prev_preference}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[Constraints]
- **NO Factual Recital**: Do NOT record specific entities (e.g., specific dates, locations, names, exact prices) or specific constraints.
- **Focus on the 'Why'**: We care about the *reasoning* behind choices, not the choices themselves.
- **Goal**: A high-quality **Behavioral User Profile** describing personality, priorities, and decision-making styles.

[PROCESSING PROTOCOL]
Analyze the input using the following steps to merge new evidence with previous memory.

**STEP 1: ATTRIBUTE ABSTRACTION (The 'Why')**
- Analyze the choices made in the Dialogue and API Calls to find the **Optimization Criteria**.
- **Detach** the specific value from the domain and identify the **Attribute Dimension** (e.g., Cost, Comfort, Speed, Risk, Effort) and **Direction** (Min/Max/Balance).
- *Example*: If the user chose a long layover to save money, record "Prioritizes cost savings over travel time," NOT "Chose a flight with a layover."
- **Inference**: Identify if the user seeks efficiency, luxury, safety, or novelty based on their selections.

**STEP 2: CROSS-DOMAIN GENERALIZATION (The 'Prediction')**
- Formulate behavioral traits using **Domain-Agnostic Terms**.
- Update the `Previous Preference` by reinforcing confirmed traits or adding new observations.
- Create sentences that predict how the user would behave in a completely different situation based on the observed attribute preferences.
- *Example logic*: If a user optimized for X in Domain A, state that "User prioritizes X across services."

[OUTPUT FORMAT]
Return ONLY a JSON object. The value must be a **text string**.
{{
    "preference": "A descriptive summary of inferred behavioral traits and decision-making styles. (e.g., 'The user consistently prioritizes efficiency and reliability over cost. They tend to avoid high-friction interactions...')"
}}
"""