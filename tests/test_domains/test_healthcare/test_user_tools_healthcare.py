"""Tests for the healthcare user tools module."""

import unittest

from tau2.domains.healthcare.environment import get_environment


class TestHealthcareUserTools(unittest.TestCase):
    """Test cases for the healthcare user tools module."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = get_environment()
        self.user_tools = self.env.user_tools

    # =========================================================================
    # Insurance and Identity Tests
    # =========================================================================

    def test_check_insurance_card(self):
        """Test checking insurance card."""
        result = self.user_tools.check_insurance_card()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Provider", result)
        self.assertIn("Policy Number", result)

    def test_confirm_identity(self):
        """Test confirming patient identity."""
        result = self.user_tools.confirm_identity()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(
            "patient" in result.lower()
            or "name" in result.lower()
            or "verification" in result.lower()
        )

    # =========================================================================
    # Symptom and Vital Signs Tests
    # =========================================================================

    def test_check_symptoms(self):
        """Test checking symptoms."""
        result = self.user_tools.check_symptoms()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_take_temperature(self):
        """Test taking temperature."""
        result = self.user_tools.take_temperature()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(
            "°" in result
            or "degrees" in result.lower()
            or "temperature" in result.lower()
        )

    def test_describe_pain(self):
        """Test describing pain."""
        result = self.user_tools.describe_pain()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_check_symptom_severity(self):
        """Test checking symptom severity."""
        result = self.user_tools.check_symptom_severity()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    # =========================================================================
    # Medication Management Tests
    # =========================================================================

    def test_check_medication_bottle(self):
        """Test checking medication bottle."""
        result = self.user_tools.check_medication_bottle()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_check_medication_bottle_specific(self):
        """Test checking specific medication."""
        result = self.user_tools.check_medication_bottle(medication_name="Lisinopril")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_authorize_pharmacy_transfer(self):
        """Test authorizing pharmacy transfer."""
        result = self.user_tools.authorize_pharmacy_transfer(
            medication_name="Lisinopril", new_pharmacy="CVS Pharmacy - Main St"
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Transfer", result)
        self.assertIn("AUTHORIZED", result)
        self.assertEqual(len(self.user_tools.device.pharmacy_transfer_requests), 1)
        self.assertEqual(
            self.user_tools.device.pharmacy_transfer_requests[0]["medication_name"],
            "Lisinopril",
        )
        self.assertEqual(
            self.user_tools.device.pharmacy_transfer_requests[0]["new_pharmacy"],
            "CVS Pharmacy - Main St",
        )

    # =========================================================================
    # Calendar and Scheduling Tests
    # =========================================================================

    def test_check_calendar(self):
        """Test checking calendar."""
        result = self.user_tools.check_calendar()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_check_calendar_specific_date(self):
        """Test checking calendar for specific date."""
        result = self.user_tools.check_calendar(date="2024-06-01")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    # =========================================================================
    # Patient Portal Tests
    # =========================================================================

    def test_open_patient_portal(self):
        """Test opening patient portal."""
        result = self.user_tools.open_patient_portal()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    # =========================================================================
    # Home Monitoring Device Tests
    # =========================================================================

    def test_measure_blood_pressure(self):
        """Test measuring blood pressure."""
        result = self.user_tools.measure_blood_pressure()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_measure_blood_glucose(self):
        """Test measuring blood glucose."""
        result = self.user_tools.measure_blood_glucose()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_measure_oxygen_saturation(self):
        """Test measuring oxygen saturation."""
        result = self.user_tools.measure_oxygen_saturation()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    # =========================================================================
    # Consent and Confirmation Tests
    # =========================================================================

    def test_provide_consent(self):
        """Test providing consent."""
        result = self.user_tools.provide_consent(consent_type="treatment")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Consent", result)
        self.assertIn("AUTHORIZED", result)
        self.assertIn("treatment", self.user_tools.device.consents_provided)

    def test_provide_consent_duplicate(self):
        """Test providing same consent twice."""
        self.user_tools.provide_consent(consent_type="telehealth")
        result = self.user_tools.provide_consent(consent_type="telehealth")
        self.assertIn("already provided consent", result)

    def test_acknowledge_instructions(self):
        """Test acknowledging instructions."""
        result = self.user_tools.acknowledge_instructions(instruction_type="medication")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Instructions Acknowledged", result)
        self.assertIn("UNDERSTOOD", result)
        self.assertIn("medication", self.user_tools.device.acknowledged_instructions)

    def test_acknowledge_instructions_duplicate(self):
        """Test acknowledging same instructions twice."""
        self.user_tools.acknowledge_instructions(instruction_type="diet")
        result = self.user_tools.acknowledge_instructions(instruction_type="diet")
        self.assertIn("already acknowledged", result)

    def test_confirm_appointment(self):
        """Test confirming appointment."""
        result = self.user_tools.confirm_appointment(appointment_id="appt_001")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Confirmation", result)
        self.assertIn("CONFIRMED", result)
        self.assertIn("appt_001", self.user_tools.device.confirmed_appointments)

    def test_confirm_appointment_duplicate(self):
        """Test confirming same appointment twice."""
        self.user_tools.confirm_appointment(appointment_id="appt_002")
        result = self.user_tools.confirm_appointment(appointment_id="appt_002")
        self.assertIn("already confirmed", result)

    # =========================================================================
    # Payment Tests
    # =========================================================================

    def test_make_payment(self):
        """Test making payment."""
        result = self.user_tools.make_payment(amount=50, payment_method="credit_card")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Payment", result)
        self.assertIn("50", result)
        self.assertIn("APPROVED", result)

    def test_make_payment_unavailable_method(self):
        """Test making payment with unavailable method."""
        result = self.user_tools.make_payment(
            amount=25, payment_method="cryptocurrency"
        )
        self.assertIn("don't have", result)

    # =========================================================================
    # Profile Update Tests
    # =========================================================================

    def test_update_emergency_contact(self):
        """Test updating emergency contact."""
        result = self.user_tools.update_emergency_contact(
            name="John Doe", phone="555-1234", relationship="spouse"
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Emergency Contact Updated", result)
        self.assertIn("John Doe", result)
        self.assertIn("UPDATED", result)
        self.assertEqual(
            self.env.user_tools.surroundings.emergency_contact.name, "John Doe"
        )
        self.assertEqual(
            self.env.user_tools.surroundings.emergency_contact.phone, "555-1234"
        )
        self.assertEqual(
            self.env.user_tools.surroundings.emergency_contact.relationship, "spouse"
        )

    def test_enable_notification_preference(self):
        """Test enabling notification preference."""
        result = self.user_tools.enable_notification_preference(
            notification_type="email"
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Notification", result)
        self.assertIn("ENABLED", result)
        self.assertIn("email", self.user_tools.device.notification_preferences)

    def test_enable_notification_preference_duplicate(self):
        """Test enabling same notification preference twice."""
        self.user_tools.enable_notification_preference(notification_type="test_results")
        result = self.user_tools.enable_notification_preference(
            notification_type="test_results"
        )
        self.assertIn("already enabled", result)

    # =========================================================================
    # Photo Upload Tests
    # =========================================================================

    def test_upload_photo(self):
        """Test uploading a photo of symptoms."""
        result = self.user_tools.upload_photo(
            body_part="left arm", description="Red rash spreading from wrist to elbow"
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Photo Upload", result)
        self.assertIn("left arm", result)
        self.assertIn("Uploaded successfully", result)
        self.assertEqual(len(self.user_tools.device.uploaded_photos), 1)
        photo = self.user_tools.device.uploaded_photos[0]
        self.assertEqual(photo["body_part"], "left arm")
        self.assertEqual(photo["description"], "Red rash spreading from wrist to elbow")
        self.assertIn("photo_id", photo)
        self.assertIn("uploaded_at", photo)


if __name__ == "__main__":
    unittest.main()
