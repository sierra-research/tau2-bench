# Healthcare Domain

The healthcare domain simulates a customer service environment where agents help patients with appointments, prescriptions, insurance verification, test results, and chronic condition monitoring.

## Overview

The healthcare domain tests agent capabilities in:

- **Workflow compliance**: Identity verification first, insurance checks before booking
- **Clinical safety**: Specific thresholds determine when to escalate to medical staff (fever ≥103°F, BP ≥180/120, etc.)
- **Bidirectional coordination**: Agent and patient both have tools - agent guides, patient performs actions
- **Mixed evaluation**: Tasks check both correct outcomes (ENV_ASSERTION) and safe procedures (ACTION)
- **Patient personas**: Easy, None (neutral), and Hard personas with different health literacy levels

## Domain at a Glance

| Metric | Value |
|--------|-------|
| **Base Tasks** | 70 (152 full, 37 small) |
| **Intents** | 8 (appointment, prescription, monitoring, telehealth, test results, triage) |
| **Personas** | Easy (24), None (23), Hard (23) |
| **Evaluation** | Mixed (ENV + ACTION) for critical workflows |
| **Patient Records** | 3 with comprehensive medical history |
| **Agent Tools** | 18 (5 evaluated in tasks) |
| **User Tools** | 20 (evaluated via ENV_ASSERTION) |

### Task Distribution by Intent

**Base Set (tasks.json - 70 tasks):**
- `appointment_scheduling` (27) - Book/cancel/reschedule with insurance verification
- `telehealth_setup` (18) - Set up remote care with consent and instructions
- `chronic_monitoring` (15) - Monitor vitals for diabetes, hypertension, COPD
- `patient_mistake` (4) - Handle patient confusion gracefully
- `urgent_triage` (3) - Triage urgent symptoms appropriately (fever, pain, breathing)
- `critical_triage` (3) - Escalate critical conditions immediately (≥103°F, ≥180/120, <90% O2)

**Full Set (tasks_full.json - 152 tasks):**
Includes all base tasks plus additional complexity variations:
- `appointment_scheduling` (71) - Extended scenarios with more edge cases
- `telehealth_setup` (35) - Additional consent and setup variations
- `chronic_monitoring` (23) - More vital sign combinations
- `patient_mistake` (8) - More confusion scenarios
- `urgent_triage` (7) - Additional symptom presentations
- `critical_triage` (3) - Same critical escalation tests
- `prescription_refill` (2) - Basic refill scenarios
- `test_results_access` (3) - Lab results review scenarios

**Small Set (tasks_small.json - 37 tasks):**
Single-subtask tasks for quick evaluation, one per (intent × persona) combination.

## Architecture

The domain uses a bidirectional setup where both agent and patient have their own tools:

```
┌─────────────────────────────────────────────────────────┐
│                  HealthcareEnvironment                  │
├────────────────────────┬────────────────────────────────┤
│   Agent Side           │   Patient Side                 │
│   (HealthcareTools)    │   (HealthcareUserTools)        │
├────────────────────────┼────────────────────────────────┤
│ • get_patient_details  │ • check_insurance_card         │
│ • verify_insurance     │ • check_calendar               │
│ • book_appointment     │ • measure_blood_pressure       │
│ • check_test_results   │ • measure_blood_glucose        │
│ • transfer_to_nurse    │ • provide_consent              │
│ • 20+ more tools       │ • 15+ more tools               │
└────────────────────────┴────────────────────────────────┘
           ↓                           ↓
    Agent performs              Patient performs
    system actions              real-world actions
```

The agent can't directly access the patient's insurance card or vital signs. Instead, it must ask the patient to check these using user-side tools. This tests realistic coordination between agent requests and patient actions.

## Getting Started

### Quick Run

```bash
# View domain info
tau2 domain healthcare

# Test with 5 random tasks
tau2 run --domain healthcare \
         --agent-llm claude-sonnet-4-5-20250929 \
         --user-llm claude-sonnet-4-5-20250929 \
         --num-tasks 5

# Try tasks interactively (will prompt for domain and task selection)
tau2 play
```

## Example: Appointment Scheduling Flow

**Scenario**: Patient needs routine checkup appointment

**Required Workflow** (identity → insurance → availability → book):

1. **Agent**: `get_patient_details("Sarah Johnson", "1985-03-15")` → Verify identity
   - Returns: `patient_id="patient_001"` (needed for subsequent calls)

2. **Agent asks**: "Can you check your insurance card?"
   - **Patient**: `check_insurance_card()` → Returns: "BlueCross BlueShield, BC123456"

3. **Agent**: `verify_insurance_coverage(patient_id="patient_001", procedure_type="routine_checkup")` → Check coverage
   - Returns: Copay $20, covered

4. **Agent**: `check_available_time_slots(doctor_id="doc_001", date="2024-05-20")` → Find slots
   - Returns: Available times [09:00, 14:00, 16:00]

5. **Agent asks**: "Can you check your calendar for May 20th at 2 PM?"
   - **Patient**: `check_calendar()` → Returns: "May 20th is available"

6. **Agent**: `book_appointment(patient_id="patient_001", doctor_id="doc_001", appointment_type="routine_checkup", date="2024-05-20", time="14:00", reason="Annual checkup")` → Book it

**Evaluation**:
- **ENV_ASSERTION**: ✓ Appointment exists in database with correct details
- **ACTION**: ✓ Correct sequence (identity first, then insurance, then availability, then book)
- Both must pass for reward = 1.0

## Policies

### Workflow Requirements

**Multi-Step Pattern** (all workflows follow this hierarchy):
1. **Identity Verification** - Always verify patient identity first using `get_patient_details(full_name, date_of_birth)`
2. **Assessment** - Gather necessary information (insurance, symptoms, vitals)
3. **Verification** - Confirm availability, eligibility, or status
4. **Action** - Execute the requested operation only after all prerequisites met

**Mandatory Sequences:**
- Appointment booking: identity → insurance → availability → book
- Prescription refills: identity → medication check → prescription verification → insurance → refill
- Chronic monitoring: identity → measure vitals → assess → schedule/transfer
- Telehealth setup: identity → consent → emergency contact → instructions

### Clinical Safety Thresholds

**Critical Values** (immediate transfer to nurse):
| Vital Sign | Transfer Threshold | Booking Range |
|------------|-------------------|---------------|
| **Fever** | ≥103°F | 100-102.9°F |
| **Pain** | ≥7/10 (severe) | 1-6/10 (mild-moderate) |
| **Blood Pressure** | ≥180/120 mmHg | 130-179/80-119 mmHg |
| **Blood Glucose** | <70 or >250 mg/dL | 100-250 mg/dL |
| **Oxygen Saturation** | <90% | 90-95% |

**Decision Rule:**
- At or above transfer threshold → `transfer_to_nurse()` immediately
- Within booking range → `book_appointment()` for follow-up
- Normal readings → Routine follow-up scheduling

### Communication Guidelines

- State appointment details (date, time, doctor, specialty, copay) clearly after booking
- Explain next steps explicitly (when to arrive, what to bring)
- Transfer to nurse for clinical questions (interpreting test results, medication advice)
- Transfer to human agent for administrative issues (billing disputes, system errors)

## Tools Overview

### Agent-Side (HealthcareTools)

**Identity & Records:**
- `get_patient_details(full_name, date_of_birth)` - Verify and retrieve patient record
- `get_chronic_conditions(patient_id)` - View patient's chronic conditions
- `get_vital_signs_history(patient_id)` - Review past vital readings

**Appointments:**
- `book_appointment(...)` - Schedule appointment
- `cancel_appointment(appointment_id)` - Cancel existing appointment
- `check_available_time_slots(doctor_id, date)` - Find available slots
- `list_available_doctors(specialty)` - Find doctors by specialty

**Insurance & Billing:**
- `verify_insurance_coverage(patient_id, procedure_type)` - Check coverage
- `calculate_cost(patient_id, appointment_type)` - Calculate copay

**Clinical:**
- `check_test_results(patient_id)` - Access lab results
- `get_prescription_details(prescription_id)` - View prescription info
- `request_prescription_refill(patient_id, prescription_id)` - Refill prescription
- `transfer_to_nurse()` - Escalate to clinical staff

### Patient-Side (HealthcareUserTools)

**Information Access:**
- `check_insurance_card()` - View insurance provider and policy number
- `check_calendar()` - Check personal availability
- `check_medication_bottle()` - Read prescription number from bottle

**Vital Measurements:**
- `measure_blood_pressure()` - Take BP reading
- `measure_blood_glucose()` - Check blood sugar
- `measure_oxygen_saturation()` - Measure O2 saturation
- `take_temperature()` - Measure temperature
- `check_symptoms()` - Describe current symptoms

**Consent & Actions:**
- `provide_consent(consent_type)` - Give consent for telehealth/billing/data
- `acknowledge_instructions(instruction_type)` - Acknowledge medical instructions
- `update_emergency_contact()` - Update emergency contact info


## Adding Tasks

To add new healthcare tasks:

1. **Create task intent** in `src/tau2/domains/healthcare/tasks/<intent>_issues.py`
   - Follow existing patterns in `appointment_issues.py`, `prescription_issues.py`, etc.
   - Define `create_<intent>_tasks()` function that returns list of tasks

2. **Define evaluation functions** in `src/tau2/domains/healthcare/tasks/evaluation_functions.py`
   - Add `is_<intent>_fixed(env)` to check if issue is resolved
   - Add `get_<intent>_env_assertions(...)` if using ENV_ASSERTION checks
   - Centralize reusable evaluation logic here

3. **Set evaluation mode**:
   - Use `["ENV_ASSERTION", "ACTION"]` for workflows requiring specific order (appointments, prescriptions)
   - Use `["ENV_ASSERTION"]` for outcome-only checks (consent, emergency contact updates)
   - Identity verification must be first action if using ACTION mode
   - Insurance verification before booking/refilling if using ACTION mode

4. **Register in task manager** in `src/tau2/domains/healthcare/tasks/create_tasks.py`
   - Import your task creation function
   - Add to the appropriate TaskManager
   - Run `python -m src.tau2.domains.healthcare.tasks.create_tasks` to regenerate task files

5. **Update splits** in `data/tau2/domains/healthcare/split_tasks.json` if adding to train/test sets

## Testing

Run the comprehensive test suite:

```bash
# Test agent-side tools (39 tests)
pytest tests/test_domains/test_healthcare/test_tools_healthcare.py -v

# Test patient-side tools (27 tests)
pytest tests/test_domains/test_healthcare/test_user_tools_healthcare.py -v

# Test both (66 tests total)
pytest tests/test_domains/test_healthcare/ -v
```

All tests should pass (100% pass rate required).

**Note**: You may see a pytest config warning about `asyncio_default_fixture_loop_scope` - this is harmless and can be ignored (the healthcare tests don't use async).

## File Structure

```
src/tau2/domains/healthcare/
├── __init__.py                        # Domain exports
├── README.md                          # This file
├── data_model.py                      # Patient, Doctor, Appointment, Prescription models
├── user_data_model.py                 # PatientDevice, PatientSurroundings
├── environment.py                     # HealthcareEnvironment + factory
├── tools.py                           # Agent-side tools (18 tools)
├── user_tools.py                      # Patient-side tools (20 tools)
├── utils.py                           # Path constants
└── tasks/
    ├── __init__.py
    ├── const.py                       # Personas and tool grounding
    ├── create_tasks.py                # Task generation pipeline
    ├── manager.py                     # TaskManager class
    ├── utils.py                       # Task composition utilities
    ├── evaluation_functions.py        # Centralized evaluation logic
    ├── appointment_issues.py          # Appointment scheduling tasks
    ├── prescription_issues.py         # Prescription refill tasks
    ├── chronic_monitoring_issues.py   # Vital signs monitoring tasks
    ├── telehealth_issues.py           # Telehealth setup tasks
    ├── test_results_issues.py         # Lab results access tasks
    ├── urgent_triage_issues.py        # Urgent symptom triage tasks
    ├── critical_triage_issues.py      # Critical condition escalation
    └── patient_mistake_issues.py      # Patient confusion handling

data/tau2/domains/healthcare/
├── db.json                            # Patient database (3 patients)
├── user_db.json                       # Patient device state
├── policy.md                          # Agent policy (512 lines)
├── tasks.json                         # Main task set (70 tasks)
├── tasks_full.json                    # All tasks (152 tasks)
├── tasks_small.json                   # Single-intent tasks (37 tasks)
└── split_tasks.json                   # Train/dev/test splits

tests/test_domains/test_healthcare/
├── test_tools_healthcare.py           # Agent tool tests (39 tests)
└── test_user_tools_healthcare.py      # Patient tool tests (27 tests)
```

## Additional Documentation

- **Agent Policy**: Full policy with clinical thresholds and workflow requirements at `data/tau2/domains/healthcare/policy.md`
- **Data Models**: Detailed schema documentation in `src/tau2/domains/healthcare/data_model.py`
- **Task Generation**: Implementation details in `src/tau2/domains/healthcare/tasks/`