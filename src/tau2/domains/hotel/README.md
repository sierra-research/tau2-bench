# Hotel Concierge Domain

## Overview

The hotel concierge domain simulates a hotel AI agent that assists guests with various requests including:
- Information about hotel amenities (pool, gym, spa, restaurant, etc.)
- Local recommendations (restaurants, pharmacies, ATMs, transit)
- Service requests (towels, maintenance, wake-up calls, room service)
- Experience bookings (tours, dining, spa treatments, activities)

This domain tests agent capabilities in:
- **Information retrieval**: Providing accurate information about amenities, hours, and local services
- **Multi-step transactions**: Booking experiences with payment processing
- **Policy compliance**: Following cancellation policies, age restrictions, and pricing rules
- **Guest service**: Handling complaints, special requests, and routing to staff when appropriate
- **Personalization**: Applying loyalty tier benefits to bookings

## Architecture

The hotel domain follows a **dual-database architecture** separating agent-side and user-side components:

### Agent-Side (HotelDB)
- Managed by `HotelTools`
- Contains guest profiles, amenities, experiences, bookings, service requests
- Agent can read guest information and perform bookings/requests
- **Cannot** directly access or modify guest preferences (privacy/security)

### User-Side (HotelUserDB)
- Managed by `HotelUserTools`
- Contains guest context and preferences (dietary restrictions, wake-up time, etc.)
- User simulator controls their own preferences
- Preferences are private to the guest

### Environment Layer
The `HotelEnvironment` class coordinates both sides:
- Provides both agent tools and user tools
- Synchronizes state between agent and user databases
- Validates guest IDs exist in both systems

This architecture ensures:
1. **Privacy**: Agent cannot directly access sensitive guest preferences
2. **Realism**: Mimics real-world scenarios where preferences are guest-controlled
3. **User Control**: Guest simulator manages their own data
4. **Consistency**: Follows the established pattern from telecom domain

## Database Schema

### Agent-Side Database (HotelDB)

The agent-side hotel database contains the following collections:

### Guests
Each guest has:
- Basic profile (name, email, phone, guest ID)
- Current room booking (room number, dates, rate)
- Loyalty tier (platinum, gold, silver, regular) and points
- Service request and experience booking history

**Note**: Guest preferences are stored in the user-side database, not accessible to the agent.

### Amenities
Hotel facilities including:
- Pool, Gym, Spa, Restaurant, Business Center, Bar
- Each with location, operating hours, capacity, and availability status

### Experiences
Local activities available for booking:
- Categories: dining, spa, tours, activities, transportation
- Pricing, duration, available times, participant limits
- Age requirements and cancellation policies

### Service Requests
Guest requests for:
- Towels, pillows, maintenance, housekeeping, wake-up calls, room service
- Status tracking (pending, in_progress, completed, cancelled)

### Experience Bookings
Confirmed experience reservations with:
- Guest, experience, date/time, participants
- Payment information and status
- Cancellation tracking

### Staff Routes
Messages routed to hotel staff for:
- Complaints, special requests, emergencies
- Complex queries beyond agent scope

### Local Information
Points of interest near the hotel:
- Pharmacies, ATMs, restaurants, transit, coffee shops
- Addresses, distances, hours, and additional info

### User-Side Database (HotelUserDB)

The user-side database contains:

### Guest Context
- Guest ID (links to agent-side guest)
- Current location in hotel (in_room, at_pool, lobby, etc.)
- Guest preferences (private to user):
  - Dietary restrictions (vegan, gluten-free, nut allergy, etc.)
  - Wake-up time preference (HH:MM format)
  - Room temperature preference (cool, moderate, warm)
  - Pillow type preference (soft, medium, firm)
  - Special requests

## Available Tools

### Agent-Side Tools (HotelTools)

#### Read Tools

**Guest Information:**
- `get_guest_details(guest_id)` - Retrieve guest profile and booking

**Amenity Information:**
- `get_amenity_info(amenity_name)` - Get hours, location, and availability
- `check_room_service_menu()` - View menu items and pricing

**Experience Management:**
- `search_experiences(category, date, max_price)` - Find available experiences
- `get_experience_details(experience_id)` - Get complete experience info
- `get_experience_booking_details(booking_id)` - View booking details

**Local Information:**
- `get_local_info(query_type)` - Find nearby pharmacies, ATMs, restaurants, etc.

**Service Tracking:**
- `get_service_request_status(request_id)` - Check service request status

#### Write Tools

**Service Requests:**
- `create_service_request(guest_id, request_type, details)` - Create new service request
- `update_service_request_status(request_id, status, notes)` - Update request status

**Experience Booking:**
- `book_experience(guest_id, experience_id, date, time, participants, payment_method)` - Book experience
- `cancel_experience_booking(booking_id)` - Cancel booking per policy

**Staff Escalation:**
- `send_to_staff(guest_id, message_type, content)` - Route message to hotel staff

### User-Side Tools (HotelUserTools)

These tools are available to the user simulator (guest) to manage their own preferences and context:

#### Read Tools
- `view_my_preferences()` - View all current preferences
- `check_dietary_restrictions()` - Check dietary restrictions list
- `check_my_location()` - Check current location in hotel

#### Write Tools
**Preference Management:**
- `update_dietary_restrictions(restrictions)` - Replace dietary restrictions list
- `add_dietary_restriction(restriction)` - Add single dietary restriction
- `set_wake_up_time(time)` - Set preferred wake-up time (HH:MM)
- `set_room_temperature_preference(temperature)` - Set temperature (cool/moderate/warm)
- `set_pillow_type(pillow_type)` - Set pillow preference (soft/medium/firm)
- `add_special_request(request)` - Add a special request
- `clear_preferences()` - Clear all preferences

**Location Management:**
- `set_location(location)` - Update current location in hotel

## Policy Highlights

### Guest Identification
- Agents must verify guest ID before sharing personal information
- Privacy protection for booking details and loyalty information

### Service Requests
- Automated handling for simple requests (towels, wake-up calls)
- Staff escalation required for emergencies, complaints, special accommodations

### Experience Booking
- Confirmation required before processing payment
- Loyalty tier discounts applied automatically:
  - Silver: 5% off
  - Gold: 10% off
  - Platinum: 15% off
- Age restrictions must be verified
- Cancellation windows vary by experience (12-48 hours)

### Payment Methods
- Accepted: credit_card, apple_pay, google_pay, room_charge
- Payment processed immediately upon booking

### Communication Standards
- Professional, friendly, helpful tone
- One tool call at a time
- Confirm details before financial transactions
- Escalate when appropriate

## Evaluation Tasks

The domain includes 16 diverse evaluation tasks covering:

**Information Queries (Tasks 0, 3, 7, 9, 13):**
- Amenity hours and locations
- Room service menu
- Local information
- Guest booking details

**Service Requests (Tasks 1, 11, 15):**
- Extra towels/pillows
- Wake-up calls
- Maintenance issues

**Experience Booking (Tasks 2, 8, 10, 14):**
- Spa treatments and tours
- Multi-participant bookings
- Age verification
- Budget constraints
- Unavailable time handling

**Complex Scenarios (Tasks 4, 5, 6, 12):**
- Guest preference updates
- Staff escalation (special requests, complaints)
- Booking cancellations
- Policy compliance

### Evaluation Criteria

Tasks are evaluated on:
- **Actions**: Required tool calls (e.g., `get_amenity_info`, `book_experience`)
- **Communicate Info**: Key facts communicated to guest (hours, prices, confirmation numbers)
- **NL Assertions**: Policy compliance (e.g., "Agent should verify room number before sharing info")
- **Env Assertions**: Database state checks (booking created, status updated, payment processed)

## Task Splits

- **base**: All 16 tasks (default)
- **train**: 10 tasks (0, 1, 2, 3, 4, 7, 9, 11, 13, 14)
- **test**: 6 tasks (5, 6, 8, 10, 12, 15)

## Running Evaluations

### View Domain Information
```bash
tau2 domain hotel
```

### Run Sample Evaluation
```bash
tau2 run --domain hotel --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5 --num-trials 1
```

### Run Full Evaluation
```bash
tau2 run --domain hotel --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 5
```

### Interactive Testing
```bash
tau2 play --domain hotel --task-id 0
```

## Example Interaction

**Guest:** "Hi, I'd like to know the pool hours."

**Agent:** *Calls `get_amenity_info("Pool")`*

**Agent:** "The pool is located on the Rooftop (10th Floor) and is open from 6:00 AM to 10:00 PM. It's a heated outdoor pool with stunning city views. Towels and lounge chairs are provided. Is there anything else I can help you with?"

## Development Notes

### Adding New Experiences
1. Add experience to `data/tau2/domains/hotel/db.json` in the `experiences` collection
2. Include all required fields: name, category, description, location, price, duration, available_times, max_participants, cancellation_hours
3. Optionally add min_age for age restrictions

### Adding New Tasks
1. Add task to `data/tau2/domains/hotel/tasks.json`
2. Define user scenario with instructions and known/unknown info
3. Specify evaluation criteria (actions, communicate_info, nl_assertions, env_assertions)
4. Update task splits in `split_tasks.json`

### Testing
Run domain tests:
```bash
pytest tests/test_domains/test_hotel/
```

Run specific test:
```bash
pytest tests/test_domains/test_hotel/test_tools_hotel.py::TestHotelTools::test_book_experience
```

## Future Enhancements

Potential extensions to the domain:
- **Overbooking scenarios**: Handle situations where experiences are fully booked
- **Multi-language support**: Test agent's ability to handle non-English speakers
- **Loyalty program complexity**: Redemption of points, tier upgrades
- **Group bookings**: Coordinating multiple rooms and experiences
- **Accessibility requests**: Special accommodations for guests with disabilities
- **Seasonal pricing**: Dynamic pricing based on demand
- **Package deals**: Bundled experiences at discounted rates

## Related Domains

- **Airline**: Similar booking and cancellation flows
- **Retail**: Customer service and policy compliance
- **Telecom**: Multi-step transactions and account management

