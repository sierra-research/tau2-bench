from typing import List, Optional

from loguru import logger

from tau2.domains.healthcare.user_data_model import (
    HealthcareUserDB,
    InsuranceCard,
    MedicationBottle,
    PatientPortalInfo,
    Symptom,
    TimeSlot,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HealthcareUserTools(ToolKitBase):
    """Patient-accessible tools for healthcare interactions."""

    db: HealthcareUserDB

    def __init__(self, db: HealthcareUserDB) -> None:
        super().__init__(db)

    @property
    def device(self):
        """Patient device."""
        return self.db.patient_device

    @property
    def surroundings(self):
        """Patient surroundings."""
        return self.db.surroundings

    @is_tool(ToolType.READ)
    def check_insurance_card(self) -> str:
        """
        Look at your insurance card and read the information printed on it.
        This shows your insurance provider, policy number, group number, and copay information.

        Returns:
            A formatted string with all information visible on the insurance card
        """
        card = self.device.insurance_card
        result = f"""Insurance Card Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provider: {card.provider}
Member Name: {card.member_name}
Policy Number: {card.policy_number}
Group Number: {card.group_number}
Copay Info: {card.copay_info}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        logger.info(
            f"Patient checked insurance card: {card.provider} - {card.policy_number}"
        )
        return result

    @is_tool(ToolType.READ)
    def check_symptoms(self) -> str:
        """
        Describe the symptoms you are currently experiencing.
        This tells you how you're feeling right now, including severity and duration.

        Returns:
            Description of all current symptoms with severity and duration
        """
        if not self.device.current_symptoms:
            return "You are not experiencing any notable symptoms at the moment."

        result = "Current Symptoms:\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for i, symptom in enumerate(self.device.current_symptoms, 1):
            result += f"{i}. {symptom.description}\n"
            result += f"   Severity: {symptom.severity.upper()}\n"
            result += f"   Duration: {symptom.duration}\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        logger.info(
            f"Patient checked symptoms: {len(self.device.current_symptoms)} symptoms"
        )
        return result

    @is_tool(ToolType.READ)
    def take_temperature(self) -> str:
        """
        Use a thermometer to measure your current body temperature.

        Returns:
            Your current temperature reading in Fahrenheit
        """
        temp = self.device.current_temperature

        if temp is None:
            return "You don't have a thermometer available to check your temperature."

        if temp < 97.0:
            status = "below normal (hypothermia concern)"
        elif temp < 99.0:
            status = "normal"
        elif temp < 100.4:
            status = "slightly elevated"
        elif temp < 103.0:
            status = "fever"
        else:
            status = "high fever (seek immediate care)"

        result = f"Temperature Reading: {temp}°F ({status})"
        logger.info(f"Patient took temperature: {temp}°F")
        return result

    @is_tool(ToolType.READ)
    def check_medication_bottle(self, medication_name: Optional[str] = None) -> str:
        """
        Look at a medication bottle you have at home and read the label information.
        If you have multiple medications, specify which one you want to check.

        Args:
            medication_name: Optional name of specific medication to check

        Returns:
            Information printed on the medication bottle label
        """
        if not self.device.medications_at_home:
            return "You don't have any medication bottles at home."

        if medication_name:
            for med in self.device.medications_at_home:
                if medication_name.lower() in med.medication_name.lower():
                    return self._format_medication_bottle(med)
            return (
                f"You don't have a medication bottle for '{medication_name}' at home."
            )

        if len(self.device.medications_at_home) == 1:
            return self._format_medication_bottle(self.device.medications_at_home[0])

        result = (
            f"You have {len(self.device.medications_at_home)} medication bottles:\n"
        )
        for i, med in enumerate(self.device.medications_at_home, 1):
            result += f"{i}. {med.medication_name} - {med.dosage}\n"
        result += "\nSpecify which medication you want to check for full details."
        return result

    def _format_medication_bottle(self, med: MedicationBottle) -> str:
        """Format medication bottle information."""
        result = f"""Medication Bottle Label:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prescription #: {med.prescription_number}
Medication: {med.medication_name}
Dosage: {med.dosage}
Refills Remaining: {med.refills_remaining}
Prescribing Doctor: {med.prescribing_doctor}
Pharmacy: {med.pharmacy_name}
Pharmacy Phone: {med.pharmacy_phone}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        logger.info(f"Patient checked medication: {med.medication_name}")
        return result

    @is_tool(ToolType.READ)
    def check_calendar(self, date: Optional[str] = None) -> str:
        """
        Check your personal calendar to see your availability.
        If a specific date is provided, shows availability for that day only.

        Args:
            date: Optional specific date to check in YYYY-MM-DD format

        Returns:
            Your calendar availability
        """
        if not self.device.calendar_availability:
            return "Your calendar is empty - you have no scheduled conflicts."

        if date:
            slots = [s for s in self.device.calendar_availability if s.date == date]
            if not slots:
                return f"You have no conflicts on {date} - completely available."

            result = f"Your availability on {date}:\n"
            result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for slot in slots:
                status = "✓ Available" if slot.available else f"✗ Busy - {slot.reason}"
                result += f"{slot.time}: {status}\n"
            return result

        result = "Your Calendar Availability:\n"
        result += "━━━━━━━━━━━━━━━━��━━━━━━━━━━━━━━━━━━━━\n"

        dates = {}
        for slot in self.device.calendar_availability:
            if slot.date not in dates:
                dates[slot.date] = []
            dates[slot.date].append(slot)

        for date, slots in sorted(dates.items()):
            result += f"\n{date}:\n"
            for slot in slots:
                status = "✓ Available" if slot.available else f"✗ Busy - {slot.reason}"
                result += f"  {slot.time}: {status}\n"

        logger.info(f"Patient checked calendar: {len(dates)} dates")
        return result

    @is_tool(ToolType.READ)
    def open_patient_portal(self) -> str:
        """
        Log in to your patient portal online to view your health information.
        Shows upcoming appointments, recent visits, test results, messages, and billing.

        Returns:
            Summary of information available in your patient portal
        """
        if not self.surroundings.has_internet_access:
            return (
                "You don't have internet access right now to open the patient portal."
            )

        portal = self.device.portal_info
        if not portal:
            return "Unable to access patient portal. You may need to contact the office for login credentials."

        result = """Patient Portal Dashboard:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        if portal.upcoming_appointments:
            result += (
                f"\n📅 Upcoming Appointments ({len(portal.upcoming_appointments)}):\n"
            )
            for apt in portal.upcoming_appointments:
                result += f"  • {apt}\n"
        else:
            result += "\n📅 Upcoming Appointments: None scheduled\n"

        if portal.recent_visits:
            result += f"\n🏥 Recent Visits ({len(portal.recent_visits)}):\n"
            for visit in portal.recent_visits:
                result += f"  • {visit}\n"

        if portal.test_results_available and portal.test_results:
            result += f"\n🔬 Test Results ({len(portal.test_results)} available):\n"
            for test in portal.test_results:
                result += f"  • {test['test_name']} ({test['test_date']})\n"
                result += f"    Result: {test['result']}\n"
                if test.get("notes"):
                    result += f"    Notes: {test['notes']}\n"
        elif portal.test_results_available:
            result += "\n🔬 Test Results: ✓ New results available to view\n"
        else:
            result += "\n🔬 Test Results: No new results\n"

        if portal.messages_count > 0:
            result += f"\n✉️  Messages: {portal.messages_count} unread message(s)\n"
        else:
            result += "\n✉️  Messages: No new messages\n"

        if portal.outstanding_balance > 0:
            result += f"\n💰 Outstanding Balance: ${portal.outstanding_balance}\n"
        else:
            result += "\n💰 Outstanding Balance: $0 - All paid\n"

        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        logger.info("Patient opened portal")
        return result

    @is_tool(ToolType.READ)
    def confirm_identity(self) -> str:
        """
        Provide your identifying information for verification (name and date of birth).
        This is used to confirm your identity before discussing health information.

        Returns:
            Your full name and date of birth
        """
        from datetime import datetime

        try:
            dob = datetime.strptime(self.surroundings.date_of_birth, "%Y-%m-%d")
            formatted_dob = dob.strftime("%B %d, %Y")
        except (ValueError, TypeError):
            formatted_dob = self.surroundings.date_of_birth

        result = f"""Identity Verification:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full Name: {self.surroundings.full_name}
Date of Birth: {formatted_dob}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        logger.info(f"Patient confirmed identity: {self.surroundings.patient_id}")
        return result

    @is_tool(ToolType.WRITE)
    def make_payment(self, amount: int, payment_method: str = "credit_card") -> str:
        """
        Make a payment for medical services.

        Args:
            amount: Amount to pay in dollars
            payment_method: Payment method to use (credit_card, debit_card, cash)

        Returns:
            Payment confirmation
        """
        if payment_method not in self.surroundings.payment_methods_available:
            available = ", ".join(self.surroundings.payment_methods_available)
            return f"You don't have {payment_method} available. You can pay with: {available}"

        result = f"""Payment Confirmation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Amount Paid: ${amount}
Payment Method: {payment_method.replace("_", " ").title()}
Status: ✓ APPROVED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thank you for your payment!"""

        logger.info(f"Patient made payment: ${amount} via {payment_method}")
        return result

    @is_tool(ToolType.WRITE)
    def confirm_appointment(self, appointment_id: str) -> str:
        """
        Confirm that you will attend a scheduled appointment.

        Args:
            appointment_id: The appointment ID to confirm

        Returns:
            Confirmation message
        """
        if appointment_id in self.device.confirmed_appointments:
            return f"You have already confirmed appointment {appointment_id}."

        self.device.confirmed_appointments.append(appointment_id)
        logger.info(f"Patient confirmed appointment: {appointment_id}")

        return f"""Appointment Confirmation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Appointment ID: {appointment_id}
Status: ✓ CONFIRMED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thank you for confirming. We'll see you at your scheduled time!"""

    @is_tool(ToolType.WRITE)
    def provide_consent(self, consent_type: str) -> str:
        """
        Provide consent for treatment, procedures, or data sharing.

        Args:
            consent_type: Type of consent (e.g., "telehealth", "treatment", "data_sharing", "billing")

        Returns:
            Consent confirmation
        """
        if consent_type in self.device.consents_provided:
            return f"You have already provided consent for {consent_type}."

        self.device.consents_provided.append(consent_type)
        logger.info(f"Patient provided consent: {consent_type}")

        return f"""Consent Provided:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Consent Type: {consent_type}
Status: ✓ AUTHORIZED
Date: {self._get_current_date()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your consent has been recorded and is now active."""

    @is_tool(ToolType.WRITE)
    def acknowledge_instructions(self, instruction_type: str) -> str:
        """
        Acknowledge that you understand and will follow medical instructions.

        Args:
            instruction_type: Type of instructions (e.g., "medication", "pre_surgery", "post_care", "diet")

        Returns:
            Acknowledgment confirmation
        """
        if instruction_type in self.device.acknowledged_instructions:
            return f"You have already acknowledged {instruction_type} instructions."

        self.device.acknowledged_instructions.append(instruction_type)
        logger.info(f"Patient acknowledged instructions: {instruction_type}")

        return f"""Instructions Acknowledged:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: {instruction_type}
Status: ✓ UNDERSTOOD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have confirmed understanding of these instructions.
Please follow them as directed by your healthcare provider."""

    @is_tool(ToolType.WRITE)
    def update_emergency_contact(self, name: str, phone: str, relationship: str) -> str:
        """
        Update your emergency contact information on file.

        Args:
            name: Emergency contact's full name
            phone: Emergency contact's phone number
            relationship: Relationship to you (e.g., "spouse", "parent", "sibling", "friend")

        Returns:
            Update confirmation
        """
        from tau2.domains.healthcare.user_data_model import EmergencyContact

        self.surroundings.emergency_contact = EmergencyContact(
            name=name, phone=phone, relationship=relationship
        )
        logger.info(f"Patient updated emergency contact: {name} ({relationship})")

        return f"""Emergency Contact Updated:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {name}
Phone: {phone}
Relationship: {relationship}
Status: ✓ UPDATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your emergency contact information has been updated."""

    @is_tool(ToolType.WRITE)
    def enable_notification_preference(self, notification_type: str) -> str:
        """
        Enable appointment reminders, test result alerts, or prescription refill reminders.

        Args:
            notification_type: Type of notification ("appointment_reminders", "test_results",
                             "refill_reminders", "health_alerts")

        Returns:
            Notification preference confirmation
        """
        if notification_type in self.device.notification_preferences:
            return f"{notification_type} notifications are already enabled."

        self.device.notification_preferences.append(notification_type)
        logger.info(f"Patient enabled notification: {notification_type}")

        return f"""Notification Preference Updated:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Notification Type: {notification_type}
Status: ✓ ENABLED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You will now receive {notification_type} notifications."""

    @is_tool(ToolType.WRITE)
    def authorize_pharmacy_transfer(
        self, medication_name: str, new_pharmacy: str
    ) -> str:
        """
        Authorize transferring a prescription to a different pharmacy.

        Args:
            medication_name: Name of the medication to transfer
            new_pharmacy: Name and location of the new pharmacy

        Returns:
            Transfer authorization confirmation
        """
        transfer_request = {
            "medication_name": medication_name,
            "new_pharmacy": new_pharmacy,
            "requested_date": self._get_current_date(),
        }

        self.device.pharmacy_transfer_requests.append(transfer_request)
        logger.info(
            f"Patient requested pharmacy transfer: {medication_name} to {new_pharmacy}"
        )

        return f"""Pharmacy Transfer Request:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Medication: {medication_name}
New Pharmacy: {new_pharmacy}
Status: ✓ AUTHORIZED
━━━━━��━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your transfer request has been submitted.
The new pharmacy will contact your current pharmacy to complete the transfer.
This typically takes 1-2 business days."""

    def _get_current_date(self) -> str:
        """Helper to get current date for confirmations."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")

    # ============================================================================
    # INITIALIZATION ACTIONS - Used to set up test scenarios
    # ============================================================================

    def set_user_info(self, name: str, patient_id: str, date_of_birth: str) -> None:
        """Set the patient's identifying information for scenario initialization."""
        self.surroundings.full_name = name
        self.surroundings.patient_id = patient_id
        self.surroundings.date_of_birth = date_of_birth
        logger.info(f"Initialized user info: {name} ({patient_id})")

    def set_user_location(self, location: str) -> None:
        """Set the patient's current location."""
        self.surroundings.location = location  # type: ignore
        logger.info(f"Set user location: {location}")

    def set_insurance_info(
        self,
        provider: str,
        policy_number: str,
        group_number: str,
        member_name: str,
        copay_info: str,
    ) -> None:
        """Initialize insurance card information."""
        self.device.insurance_card = InsuranceCard(
            provider=provider,
            policy_number=policy_number,
            group_number=group_number,
            member_name=member_name,
            copay_info=copay_info,
        )
        logger.info(f"Set insurance: {provider} - {policy_number}")

    def add_medication_at_home(
        self,
        prescription_number: str,
        medication_name: str,
        dosage: str,
        refills_remaining: int,
        prescribing_doctor: str,
        pharmacy_name: str,
        pharmacy_phone: str,
    ) -> None:
        """Add a medication bottle to patient's home."""
        med = MedicationBottle(
            prescription_number=prescription_number,
            medication_name=medication_name,
            dosage=dosage,
            refills_remaining=refills_remaining,
            prescribing_doctor=prescribing_doctor,
            pharmacy_name=pharmacy_name,
            pharmacy_phone=pharmacy_phone,
        )
        self.device.medications_at_home.append(med)
        logger.info(f"Added medication: {medication_name}")

    def add_symptom(self, description: str, severity: str, duration: str) -> None:
        """Add a symptom the patient is experiencing."""
        symptom = Symptom(
            description=description,
            severity=severity,
            duration=duration,  # type: ignore
        )
        self.device.current_symptoms.append(symptom)
        logger.info(f"Added symptom: {description} ({severity})")

    def set_temperature(self, temperature: float) -> None:
        """Set the patient's current body temperature."""
        self.device.current_temperature = temperature
        logger.info(f"Set temperature: {temperature}°F")

    def add_calendar_slot(
        self, date: str, time: str, available: bool, reason: Optional[str] = None
    ) -> None:
        """Add a time slot to the patient's calendar."""
        slot = TimeSlot(date=date, time=time, available=available, reason=reason)
        self.device.calendar_availability.append(slot)
        logger.info(
            f"Added calendar slot: {date} {time} - {'Available' if available else 'Busy'}"
        )

    def set_portal_info(
        self,
        upcoming_appointments: List[str],
        recent_visits: List[str],
        test_results_available: bool,
        messages_count: int,
        outstanding_balance: float,
        test_results: Optional[List[dict]] = None,
    ) -> None:
        """Initialize patient portal information."""
        self.device.portal_info = PatientPortalInfo(
            upcoming_appointments=upcoming_appointments,
            recent_visits=recent_visits,
            test_results_available=test_results_available,
            test_results=test_results if test_results is not None else [],
            messages_count=messages_count,
            outstanding_balance=outstanding_balance,  # type: ignore
        )
        logger.info("Initialized patient portal info")

    def set_bp_monitor(self, has_monitor: bool, systolic: int, diastolic: int) -> None:
        """Set blood pressure monitor and reading."""
        from tau2.domains.healthcare.user_data_model import BloodPressureReading

        self.device.has_blood_pressure_monitor = has_monitor
        if has_monitor:
            self.device.latest_bp_reading = BloodPressureReading(
                systolic=systolic, diastolic=diastolic
            )
            logger.info(f"Set BP monitor: {systolic}/{diastolic} mmHg")

    def set_glucose_monitor(
        self, has_monitor: bool, glucose_reading: int, measurement_time: str
    ) -> None:
        """Set glucose meter and reading."""
        self.device.has_glucose_meter = has_monitor
        if has_monitor:
            self.device.latest_glucose_reading = glucose_reading
            self.device.glucose_measurement_time = measurement_time
            logger.info(
                f"Set glucose monitor: {glucose_reading} mg/dL ({measurement_time})"
            )

    def set_pulse_oximeter(self, has_monitor: bool, spo2: int, heart_rate: int) -> None:
        """Set pulse oximeter and reading."""
        self.device.has_pulse_oximeter = has_monitor
        if has_monitor:
            self.device.latest_spo2_reading = spo2
            self.device.latest_heart_rate = heart_rate
            logger.info(f"Set pulse oximeter: SpO2 {spo2}%, HR {heart_rate} bpm")

    def set_emergency_contact(self, name: str, phone: str, relationship: str) -> None:
        """Set emergency contact information."""
        from tau2.domains.healthcare.user_data_model import EmergencyContact

        self.surroundings.emergency_contact = EmergencyContact(
            name=name, phone=phone, relationship=relationship
        )
        logger.info(f"Set emergency contact: {name} ({relationship})")

    # ============================================================================
    # ENV_ASSERTION METHODS - Used for deterministic evaluation
    # ============================================================================

    def assert_has_calendar_availability(self, expected: bool) -> bool:
        """Assert whether patient has any calendar availability configured."""
        has_availability = len(self.device.calendar_availability) > 0
        return has_availability == expected

    def assert_has_insurance_card(self, expected: bool) -> bool:
        """Assert whether patient has insurance card information."""
        has_card = self.device.insurance_card is not None
        return has_card == expected

    def assert_insurance_provider(self, expected_provider: str) -> bool:
        """Assert the insurance provider name."""
        if not self.device.insurance_card:
            return False
        return self.device.insurance_card.provider == expected_provider

    def assert_has_symptoms(self, expected: bool) -> bool:
        """Assert whether patient has current symptoms."""
        has_symptoms = len(self.device.current_symptoms) > 0
        return has_symptoms == expected

    def assert_temperature_reading(self, expected_temp: float) -> bool:
        """Assert the patient's current temperature reading."""
        return self.device.current_temperature == expected_temp

    def assert_medication_count(self, expected_count: int) -> bool:
        """Assert the number of medications at home."""
        return len(self.device.medications_at_home) == expected_count

    def assert_has_portal_access(self, expected: bool) -> bool:
        """Assert whether patient has portal information available."""
        has_portal = self.device.portal_info is not None
        return has_portal == expected

    def assert_consent_provided(self, consent_type: str) -> bool:
        """Assert whether patient has provided a specific type of consent."""
        return consent_type in self.device.consents_provided

    def assert_instructions_acknowledged(self, instruction_type: str) -> bool:
        """Assert whether patient has acknowledged a specific type of instructions."""
        return instruction_type in self.device.acknowledged_instructions

    def assert_emergency_contact_updated(self, name: str, relationship: str) -> bool:
        """Assert whether emergency contact has been updated with specific details."""
        if not self.surroundings.emergency_contact:
            return False
        contact = self.surroundings.emergency_contact
        return contact.name == name and contact.relationship == relationship

    # ============================================================================
    # PATIENT MEDICAL MONITORING TOOLS - Home health measurements
    # ============================================================================

    @is_tool(ToolType.READ)
    def measure_blood_pressure(self) -> str:
        """
        Use a home blood pressure monitor to measure blood pressure.

        Returns:
            Blood pressure reading with systolic/diastolic values
        """
        if (
            not hasattr(self.device, "has_blood_pressure_monitor")
            or not self.device.has_blood_pressure_monitor
        ):
            return "You don't have a blood pressure monitor at home."

        # Get simulated reading from device
        reading = self.device.latest_bp_reading

        if reading is None:
            return "You haven't taken your blood pressure yet. Please take a measurement first."

        systolic = reading.systolic
        diastolic = reading.diastolic

        # Interpret the reading
        if systolic >= 180 or diastolic >= 120:
            status = "⚠️ HYPERTENSIVE CRISIS - Seek immediate medical attention"
        elif systolic >= 140 or diastolic >= 90:
            status = "High (Stage 2 Hypertension)"
        elif systolic >= 130 or diastolic >= 80:
            status = "Elevated (Stage 1 Hypertension)"
        else:
            status = "Normal"

        result = f"""Blood Pressure Reading:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Systolic: {systolic} mmHg
Diastolic: {diastolic} mmHg
Result: {systolic}/{diastolic} mmHg
Status: {status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        logger.info(f"Patient measured BP: {systolic}/{diastolic}")
        return result

    @is_tool(ToolType.READ)
    def measure_blood_glucose(self) -> str:
        """
        Use a glucometer to measure blood glucose level.

        Returns:
            Blood glucose reading in mg/dL
        """
        if (
            not hasattr(self.device, "has_glucose_meter")
            or not self.device.has_glucose_meter
        ):
            return "You don't have a glucose meter at home."

        reading = self.device.latest_glucose_reading  # mg/dL

        if reading is None:
            return "You haven't measured your blood glucose yet. Please take a measurement first."

        # Interpret reading (simplified - depends on fasting or not)
        if reading < 70:
            status = "⚠️ LOW (Hypoglycemia) - Consume fast-acting carbs immediately"
        elif reading <= 100:
            status = "Normal (fasting)"
        elif reading <= 125:
            status = "Elevated (Prediabetes range)"
        else:
            status = "⚠️ HIGH (Diabetes range)"

        result = f"""Blood Glucose Reading:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Glucose: {reading} mg/dL
Status: {status}
Time: {self.device.glucose_measurement_time}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        logger.info(f"Patient measured glucose: {reading} mg/dL")
        return result

    @is_tool(ToolType.READ)
    def measure_oxygen_saturation(self) -> str:
        """
        Use a pulse oximeter to measure blood oxygen saturation (SpO2).

        Returns:
            SpO2 percentage and heart rate
        """
        if (
            not hasattr(self.device, "has_pulse_oximeter")
            or not self.device.has_pulse_oximeter
        ):
            return "You don't have a pulse oximeter at home."

        spo2 = self.device.latest_spo2_reading
        heart_rate = self.device.latest_heart_rate

        if spo2 is None or heart_rate is None:
            return "You haven't measured your oxygen saturation yet. Please take a measurement first."

        # Interpret reading
        if spo2 < 85:
            status = "⚠️ SEVERE HYPOXEMIA - Seek immediate medical attention"
        elif spo2 < 90:
            status = "Moderate Hypoxemia - Contact doctor"
        elif spo2 < 95:
            status = "Mild Hypoxemia - Monitor closely"
        else:
            status = "Normal"

        result = f"""Pulse Oximeter Reading:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SpO2: {spo2}%
Heart Rate: {heart_rate} bpm
Status: {status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        logger.info(f"Patient measured SpO2: {spo2}%, HR: {heart_rate}")
        return result

    @is_tool(ToolType.READ)
    def describe_pain(self) -> str:
        """
        Describe current pain using standardized PQRST assessment.

        Returns:
            Pain assessment with provocation, quality, radiation, severity, and timing
        """
        if not hasattr(self.device, "current_pain") or not self.device.current_pain:
            return "You are not experiencing any significant pain right now."

        pain = self.device.current_pain

        result = f"""Pain Assessment (PQRST):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P - Provocation: {pain.provocation}
Q - Quality: {pain.quality}
R - Radiation: {pain.radiation}
S - Severity: {pain.severity}/10
T - Timing: {pain.timing}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        logger.info(f"Patient describing pain: severity {pain.severity}/10")
        return result

    @is_tool(ToolType.WRITE)
    def upload_photo(self, body_part: str, description: str) -> str:
        """
        Upload a photo of symptoms (rash, wound, swelling, etc.).
        Used in telehealth for visual assessment.

        Args:
            body_part: Location of symptom (e.g., "left arm", "abdomen", "face")
            description: Brief description of what's shown

        Returns:
            Confirmation of photo upload
        """
        from datetime import datetime

        photo_id = f"PHOTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store photo metadata
        photo_record = {
            "photo_id": photo_id,
            "body_part": body_part,
            "description": description,
            "uploaded_at": datetime.now().isoformat(),
        }
        self.device.uploaded_photos.append(photo_record)

        logger.info(f"Patient uploaded photo: {body_part} - {description}")

        return f"""Photo Upload Confirmation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Photo ID: {photo_id}
Body Part: {body_part}
Description: {description}
Status: ✓ Uploaded successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The doctor will review this photo and may request additional views if needed."""

    @is_tool(ToolType.READ)
    def check_symptom_severity(self) -> str:
        """
        Assess the overall severity of current symptoms.
        Helps determine if urgent care is needed.

        Returns:
            Summary of symptoms with severity assessment
        """
        if (
            not hasattr(self.device, "current_symptoms")
            or not self.device.current_symptoms
        ):
            return "You are not experiencing any symptoms at this time."

        symptoms = self.device.current_symptoms

        # Count by severity
        severe = [s for s in symptoms if s.severity == "severe"]
        moderate = [s for s in symptoms if s.severity == "moderate"]
        mild = [s for s in symptoms if s.severity == "mild"]

        result = f"""Symptom Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Symptoms: {len(symptoms)}
Severe: {len(severe)}
Moderate: {len(moderate)}
Mild: {len(mild)}

Details:
"""
        for symptom in symptoms:
            result += f"- {symptom.description} ({symptom.severity}) - Duration: {symptom.duration}\n"

        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if severe:
            result += "\n⚠️ You have severe symptoms. Consider seeking immediate medical attention."

        logger.info(
            f"Patient checked symptoms: {len(symptoms)} total, {len(severe)} severe"
        )
        return result
