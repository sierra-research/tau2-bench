# Food Domain Voice Audio Assets (`16 kHz`, Mono, `PCM_S16LE`)

All food-domain ambient beds, out-of-domain background chatter, and in-domain store-matched distractor clips live under `data/voice/background_noise_audio_pcm_mono_verified/` and are wired into `food_{store}_{channel}` presets by [`voice_presets.py`](voice_presets.py). Every clip is `16 kHz` mono `16-bit PCM` (`.wav`) audio generated via `gemini-3.8-flash-tts` (including synthesized instrumental grooves and multi-speaker voices disjoint from primary caller personas, with zero external music samples or brand trademarks) and verified via programmatic and multimodal QA.

## 1. Continuous Ambient Beds (`data/voice/background_noise_audio_pcm_mono_verified/continuous/`)
- Shared universal beds reused by food presets (`continuous/*.wav`):
  - **Drive-thru**: `busy_street_iphone_mic.wav`
  - **Mobile**: `people_talking.wav`, `medium_size_room_tv_news_iphone_mic.wav`
- Food-domain continuous beds (`continuous/food/*.wav` — `3` clips):
  - `food/car_cabin_radio_music.wav` — Loopable in-cabin car audio instrumental groove for drive-thru environments.
  - `food/car_cabin_radio_talk.wav` — Two-host FM radio afternoon drive and traffic report over a synthesized instrumental bed, followed by the station's music, for drive-thru environments. Re-rendered as heard from inside the caller's idling car through the open window (car-speaker band 150–3000 Hz, cabin reverb RT60 0.25 s at −2 dB direct-to-reverberant, 2.6 kHz occlusion low-pass, light engine idle 6 dB below the voice); 49.7 s seamless loop.
  - `food/living_room_tv_broadcast.wav` — Two-anchor local television news and weekend weather broadcast, followed by an instrumental break, for indoor/mobile environments. Re-rendered as a TV across a living room picked up by the caller's phone (TV-speaker band 180–6000 Hz, room reverb RT60 0.55 s at −6 dB direct-to-reverberant, 3.8 kHz off-axis low-pass, room tone at the voice level); 37.9 s seamless loop.
  - Beds are RMS-normalized per 200 ms chunk when mixed, so a clean studio voice would always sit at the full bed level; the far-field rendering and steady ambience keep the broadcast voice below the bed level so the agent does not mistake it for the caller.

## 2. Pure Environmental Acoustic Bursts (`data/voice/background_noise_audio_pcm_mono_verified/bursts/*.wav` — `burst_noise` Stream)
Non-speech environmental sounds scheduled on `burst_noise`:
- **Drive-thru (`3` clips)**: `car_horn.wav`, `engine_idling.wav`, `siren.wav`.
- **Mobile (`2` clips)**: `ringing_phone.wav`, `dog_bark.wav`.

## 3. Secondary-Speaker Background Speech (`data/voice/background_noise_audio_pcm_mono_verified/bursts/food/**` — `background_speech` Stream)
All secondary-speaker utterances (`34` WAVs total) are scheduled on an independent `background_speech` stream using a deterministic 10-slot `background_speech_files` pool (`3` Out-of-Domain chatter slots + `7` gold-disjoint In-Domain distractor slots under uniform `EffectScheduler` selection) so every food task receives a consistent 70% / 30% in-domain-to-out-of-domain mix regardless of how many items conflict with the gold order.

### 3a. Out-of-Domain Background Chatter (`bursts/food/out_of_domain/*.wav` — `3 / 10` Pool Slots = `30%`)
Off-axis passenger and household chatter with zero restaurant menu or reservation commands (`7` clips, no per-task conflict filtering needed):
- **Drive-thru car cabin (`3` clips)**:
  - `food/out_of_domain/side_speech_seatbelt.wav` — Off-axis parent reminding backseat child to buckle seatbelt.
  - `food/out_of_domain/side_speech_radio_down.wav` — Passenger asking to turn the car radio down.
  - `food/out_of_domain/side_speech_car_chat_exit.wav` — Two passengers chatting about taking the coast exit past the traffic light.
- **Mobile household (`4` clips)**:
  - `food/out_of_domain/side_speech_sit_down.wav` — Off-axis parent telling a child to sit down on the couch.
  - `food/out_of_domain/side_speech_quiet.wav` — Family member shushing nearby household chatter while someone is on the phone.
  - `food/out_of_domain/side_speech_family_chat.wav` — Two family members across the kitchen talking about keys by the toaster.
  - `food/out_of_domain/side_speech_roommate_plates.wav` — Two roommates checking on paper plates and napkins at the kitchen counter.

### 3b. In-Domain Store & Channel Distractors (`bursts/food/in_domain/*.wav` — `7 / 10` Pool Slots = `70%`)
Semantic speaker-attribution distractors (`27` clips total, bandpass-thinned to `700–2500 Hz` and filtered per task against gold order items and reservation actions):
- **Drive-thru adjacent-lane QSR exchanges (`15` clips, `5` per store)**:
  - **Corner Cafe (`cafe`)**:
    - `food/in_domain/other_lane_cafe_mocha.wav` — *"Medium Iced Hazelnut Mocha and a Butter Croissant"*
    - `food/in_domain/other_lane_cafe_panini.wav` — *"Roast Beef Panini and a medium Lemonade"*
    - `food/in_domain/other_lane_cafe_matcha.wav` — *"Iced Matcha Latte with oat milk and a Butter Croissant"*
    - `food/in_domain/other_lane_cafe_cold_brew.wav` — *"Large Cold Brew and one Blueberry Muffin"*
    - `food/in_domain/other_lane_cafe_window.wav` — Barista and driver wrapping up at the pickup window.
  - **City Pizza (`pizza`)**:
    - `food/in_domain/other_lane_pizza_sticks.wav` — *"Mozzarella Sticks and a Bottled Cola"*
    - `food/in_domain/other_lane_pizza_ziti.wav` — *"One Baked Ziti and two Cannoli"*
    - `food/in_domain/other_lane_pizza_classic.wav` — *"One Large Classic Pie with extra cheese and Half Dozen Garlic Knots"*
    - `food/in_domain/other_lane_pizza_wings.wav` — *"Ten Buffalo Wings with a Side of Ranch"*
    - `food/in_domain/other_lane_pizza_window.wav` — Attendant directing driver to the first pickup window.
  - **Sunset Burger (`burger`)**:
    - `food/in_domain/other_lane_burger_pier.wav` — *"Pier Burger Combo and swap the drink to a Regular Banana Shake"*
    - `food/in_domain/other_lane_burger_wrap.wav` — *"Grilled Chicken Wrap and a 4 Piece Chicken Nuggets"*
    - `food/in_domain/other_lane_burger_cheeseburgers.wav` — *"Two Cheeseburgers and a Medium Fries"*
    - `food/in_domain/other_lane_burger_shake.wav` — *"One Regular Vanilla Shake and 6 Piece Chicken Nuggets"*
    - `food/in_domain/other_lane_burger_window.wav` — Attendant directing driver to the first pickup window.
- **Mobile household order & reservation chime-ins (`12` clips, `4` per store)**:
  - **Corner Cafe (`cafe`)**:
    - `food/in_domain/household_cafe_mocha.wav` — *"Hey, grab me an Iced Hazelnut Mocha while you're ordering!"*
    - `food/in_domain/household_cafe_turkey.wav` — *"Hey, get me a Turkey Club Sandwich and Kettle Chips!"*
    - `food/in_domain/household_cafe_chai.wav` — *"Grab me a hot Chai Latte and a Chocolate Chip Cookie..."*
    - `food/in_domain/household_cafe_table.wav` — *"Hey, ask if they have an outdoor table for four at seven tonight."* (dropped for reservation tasks)
  - **City Pizza (`pizza`)**:
    - `food/in_domain/household_pizza_ziti.wav` — *"Hey, tell them to add a Baked Ziti and a Cannoli too!"*
    - `food/in_domain/household_pizza_calzone.wav` — *"Hey, grab me a Cheese Calzone and a Side of Marinara!"*
    - `food/in_domain/household_pizza_hero.wav` — *"Hey, throw in a Chicken Parm Hero and a Bottled Lemonade for me."*
    - `food/in_domain/household_pizza_table.wav` — *"Ask if they have an indoor table for six around six thirty."* (dropped for reservation tasks)
  - **Sunset Burger (`burger` — ordering only, no reservations)**:
    - `food/in_domain/household_burger_fish.wav` — *"Hey, ask them to add a Crispy Fish Sandwich and Apple Slices!"*
    - `food/in_domain/household_burger_sundae.wav` — *"Hey, can we add a Fudge Brownie Sundae and a Large Fries?"*
    - `food/in_domain/household_burger_wrap.wav` — *"Could you throw in a Grilled Chicken Wrap and a Medium Root Beer?"*
    - `food/in_domain/household_burger_ranch.wav` — *"Make sure mine has no onions and Ranch Dipping Sauce on the side."*
