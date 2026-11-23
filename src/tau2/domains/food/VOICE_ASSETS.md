# Food Domain Voice Audio Assets (`16 kHz`, Mono, `PCM_S16LE`)

All food-domain ambient beds, out-of-domain background chatter, and in-domain store-matched distractor clips live under `data/voice/background_noise_audio_pcm_mono_verified/` and are wired into `food_{store}_{channel}` presets by [`voice_presets.py`](voice_presets.py). Every clip is `16 kHz` mono `16-bit PCM` (`.wav`) audio using synthesized multi-speaker TTS voices (disjoint from primary caller personas) and synthesized instrumental grooves (with zero external music samples or brand trademarks), verified via programmatic and multimodal QA.

## 1. Continuous Ambient Beds (`data/voice/background_noise_audio_pcm_mono_verified/continuous/`)
- Shared universal beds reused by food presets (`continuous/*.wav`):
  - **Drive-thru**: `busy_street_iphone_mic.wav`
  - **Mobile**: `people_talking.wav`, `medium_size_room_tv_news_iphone_mic.wav`, `street_and_metro_station_iphone_mic.wav`
- Food-domain continuous beds (`continuous/food/*.wav` — `3` clips):
  - `food/car_cabin_radio_music.wav` — Street traffic base (`busy_street_iphone_mic.wav` with `<155 Hz` rumble trimmed) fused with intermittent (`~13–15.5 s` active, `~6–8 s` street-only gap) in-cabin FM music radio swells featuring synthesized grooves, brief station DJ bumpers, and sung vocal hooks (`180.9 s`).
  - `food/car_cabin_radio_talk.wav` — Street traffic base (`busy_street_iphone_mic.wav` with `<155 Hz` rumble trimmed) fused with intermittent two-host FM drive-time traffic, weather, and sports segments rendered through a car-cabin speaker (`230–3500 Hz`, `RT60 = 0.22 s`) with `~5–7.4 s` street-only gaps (`180.9 s`).
  - `food/living_room_tv_broadcast.wav` — Diffused indoor household room ambience fused with intermittent far-field television game-show, news, sports, and documentary segments (`210–3550 Hz`, `RT60 = 0.34 s`) separated by `~4.5–6.9 s` room-only gaps (`150.0 s`).
  - Because `tau2` normalizes continuous beds per `200 ms` chunk before `300–3400 Hz` telephony filtering, trimming sub-`155 Hz` waste and pairing far-field speaker reflections with natural street/room gaps keeps `1–3.4 kHz` interference aligned with `medium_size_room_tv_news_iphone_mic.wav` without overpowering the primary caller.

## 2. Pure Environmental Acoustic Bursts (`data/voice/background_noise_audio_pcm_mono_verified/bursts/*.wav` — `burst_noise` Stream)
Non-speech environmental sounds scheduled on `burst_noise`:
- **Drive-thru (`3` clips)**: `car_horn.wav`, `engine_idling.wav`, `siren.wav`.
- **Mobile (`2` clips)**: `ringing_phone.wav`, `dog_bark.wav`.

## 3. Secondary-Speaker Background Speech (`data/voice/background_noise_audio_pcm_mono_verified/bursts/food/**` — `background_speech` Stream)
All secondary-speaker utterances (`34` WAVs total) are scheduled on an independent `background_speech` stream (`2.0 events/min` in regular mode, `3.0 events/min` in heavy channel mode, at `3–7 dB` SNR) using a deterministic 10-slot `background_speech_files` pool (`3` Out-of-Domain chatter slots + `7` gold-disjoint In-Domain distractor slots under uniform `EffectScheduler` selection) so every food task receives a consistent 70% / 30% in-domain-to-out-of-domain mix regardless of how many items conflict with the gold order.

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
