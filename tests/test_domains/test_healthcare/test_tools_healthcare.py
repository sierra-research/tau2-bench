"""Tests for the healthcare tools module."""

import unittest
from pathlib import Path

from tau2.domains.healthcare.data_model import HealthcareDB
from tau2.domains.healthcare.tools import HealthcareTools

# Path to the healthcare database file
HEALTHCARE_DB_PATH = (
    Path(__file__).parents[3] / "data" / "tau2" / "domains" / "healthcare" / "db.json"
)


class TestHealthcareTools(unittest.TestCase):
    """Test cases for the healthcare tools module."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Load the healthcare database
        self.db = HealthcareDB.load(str(HEALTHCARE_DB_PATH))
        # Create the healthcare tools instance
        self.tools = HealthcareTools(self.db)

    def test_db_loaded(self):
        """Test that the database is loaded correctly."""
        self.assertIsNotNone(self.db)
        self.assertTrue(len(self.db.patients) > 0)
        self.assertTrue(len(self.db.doctors) > 0)
        self.assertTrue(len(self.db.appointments) > 0)
        self.assertTrue(len(self.db.prescriptions) > 0)

    # =========================================================================
    # Patient Lookup Tests
    # =========================================================================

    def test_get_patient_details_success(self):
        """Test getting patient details with valid name and DOB."""
        patient = self.tools.get_patient_details(
            full_name="Sarah Johnson", date_of_birth="1985-03-15"
        )
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "patient_001")
        self.assertEqual(patient.name.first_name, "Sarah")
        self.assertEqual(patient.name.last_name, "Johnson")

    def test_get_patient_details_not_found(self):
        """Test getting patient with non-existent name."""
        with self.assertRaises(ValueError) as context:
            self.tools.get_patient_details(
                full_name="Nonexistent Person", date_of_birth="1900-01-01"
            )
        self.assertIn("no patient found", str(context.exception).lower())

    def test_get_patient_details_wrong_dob(self):
        """Test getting patient with correct name but wrong DOB."""
        with self.assertRaises(ValueError) as context:
            self.tools.get_patient_details(
                full_name="Sarah Johnson", date_of_birth="1990-01-01"
            )
        self.assertIn("no patient found", str(context.exception).lower())

    # =========================================================================
    # Doctor Lookup Tests
    # =========================================================================

    def test_list_available_doctors_all(self):
        """Test listing all available doctors."""
        doctors = self.tools.list_available_doctors()
        self.assertGreater(len(doctors), 0)
        for doctor in doctors:
            self.assertHasAttr(doctor, "doctor_id")
            self.assertHasAttr(doctor, "name")
            self.assertHasAttr(doctor, "specialty")

    def test_list_available_doctors_by_specialty(self):
        """Test listing doctors filtered by specialty."""
        doctors = self.tools.list_available_doctors(specialty="General Practice")
        self.assertGreater(len(doctors), 0)
        for doctor in doctors:
            self.assertEqual(doctor.specialty, "General Practice")

    def test_list_available_doctors_specialty_not_found(self):
        """Test listing doctors with non-existent specialty."""
        doctors = self.tools.list_available_doctors(specialty="Nonexistent Specialty")
        self.assertEqual(len(doctors), 0)

    # =========================================================================
    # Appointment Lookup Tests
    # =========================================================================

    def test_get_appointment_details_success(self):
        """Test getting appointment details with valid ID."""
        # Use first appointment from DB
        appointment_id = list(self.db.appointments.keys())[0]
        appointment = self.tools.get_appointment_details(appointment_id)
        self.assertIsNotNone(appointment)
        self.assertEqual(appointment.appointment_id, appointment_id)

    def test_get_appointment_details_not_found(self):
        """Test getting appointment with non-existent ID."""
        with self.assertRaises(ValueError) as context:
            self.tools.get_appointment_details("nonexistent_appointment")
        self.assertIn("not found", str(context.exception).lower())

    def test_search_appointments_by_patient(self):
        """Test searching appointments for a specific patient."""
        appointments = self.tools.search_appointments(patient_id="patient_001")
        self.assertIsInstance(appointments, list)
        for appt in appointments:
            self.assertEqual(appt.patient_id, "patient_001")

    def test_search_appointments_by_status(self):
        """Test searching appointments by status."""
        appointments = self.tools.search_appointments(
            patient_id="patient_001", status="scheduled"
        )
        self.assertIsInstance(appointments, list)
        for appt in appointments:
            self.assertEqual(appt.status, "scheduled")

    def test_search_appointments_patient_not_found(self):
        """Test searching appointments for non-existent patient."""
        with self.assertRaises(ValueError):
            self.tools.search_appointments(patient_id="nonexistent_patient")

    # =========================================================================
    # Appointment Time Slot Tests
    # =========================================================================

    def test_check_available_time_slots_success(self):
        """Test checking available time slots for a doctor."""
        doctor_id = list(self.db.doctors.keys())[0]
        slots = self.tools.check_available_time_slots(
            doctor_id=doctor_id, date="2024-06-15"
        )
        self.assertIsInstance(slots, list)
        for slot in slots:
            self.assertIn("time", slot)
            self.assertIn("available", slot)

    def test_check_available_time_slots_doctor_not_found(self):
        """Test checking time slots for non-existent doctor."""
        with self.assertRaises(ValueError):
            self.tools.check_available_time_slots(
                doctor_id="nonexistent_doctor", date="2024-06-15"
            )

    # =========================================================================
    # Appointment Booking Tests
    # =========================================================================

    def test_book_appointment_success(self):
        """Test booking a new appointment."""
        result = self.tools.book_appointment(
            patient_id="patient_001",
            doctor_id="doc_001",
            appointment_type="routine_checkup",
            date="2024-12-02",
            time="14:00",
            reason="Annual checkup",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "scheduled")
        self.assertEqual(result.patient_id, "patient_001")
        self.assertEqual(result.doctor_id, "doc_001")
        self.assertIn(result.appointment_id, self.db.appointments)

    def test_book_appointment_patient_not_found(self):
        """Test booking appointment for non-existent patient."""
        with self.assertRaises(ValueError):
            self.tools.book_appointment(
                patient_id="nonexistent_patient",
                doctor_id="doc_001",
                appointment_type="routine_checkup",
                date="2024-12-02",
                time="14:00",
                reason="Test",
            )

    def test_book_appointment_doctor_not_found(self):
        """Test booking appointment with non-existent doctor."""
        with self.assertRaises(ValueError):
            self.tools.book_appointment(
                patient_id="patient_001",
                doctor_id="nonexistent_doctor",
                appointment_type="routine_checkup",
                date="2024-12-02",
                time="14:00",
                reason="Test",
            )

    # =========================================================================
    # Appointment Modification Tests
    # =========================================================================

    def test_cancel_appointment_success(self):
        """Test canceling an existing appointment."""
        booking_result = self.tools.book_appointment(
            patient_id="patient_001",
            doctor_id="doc_001",
            appointment_type="routine_checkup",
            date="2024-12-02",
            time="10:00",
            reason="Test appointment for cancellation",
        )
        appointment_id = booking_result.appointment_id

        result = self.tools.cancel_appointment(
            appointment_id=appointment_id, reason="Patient canceled"
        )
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.appointment_id, appointment_id)

    def test_cancel_appointment_not_found(self):
        """Test canceling non-existent appointment."""
        with self.assertRaises(ValueError):
            self.tools.cancel_appointment(
                appointment_id="nonexistent_appointment", reason="Test"
            )

    def test_reschedule_appointment_success(self):
        """Test rescheduling an existing appointment."""
        booking_result = self.tools.book_appointment(
            patient_id="patient_001",
            doctor_id="doc_001",
            appointment_type="routine_checkup",
            date="2024-12-03",
            time="10:00",
            reason="Test appointment for rescheduling",
        )
        appointment_id = booking_result.appointment_id

        result = self.tools.reschedule_appointment(
            appointment_id=appointment_id, new_date="2024-12-10", new_time="15:00"
        )
        self.assertEqual(result.appointment_id, appointment_id)
        self.assertEqual(result.date, "2024-12-10")
        self.assertEqual(result.time, "15:00")

    def test_reschedule_appointment_not_found(self):
        """Test rescheduling non-existent appointment."""
        with self.assertRaises(ValueError):
            self.tools.reschedule_appointment(
                appointment_id="nonexistent_appointment",
                new_date="2024-12-10",
                new_time="15:00",
            )

    # =========================================================================
    # Insurance Tests
    # =========================================================================

    def test_verify_insurance_coverage_success(self):
        """Test verifying insurance coverage for a patient."""
        result = self.tools.verify_insurance_coverage(
            patient_id="patient_001", procedure_type="routine_checkup"
        )
        self.assertIn("verified", result)
        self.assertIn("provider", result)
        self.assertIn("policy_number", result)
        self.assertIn("copay_amount", result)
        self.assertIn("coverage_details", result)
        self.assertIn("procedure_covered", result)

    def test_verify_insurance_coverage_patient_not_found(self):
        """Test verifying insurance for non-existent patient."""
        with self.assertRaises(ValueError):
            self.tools.verify_insurance_coverage(
                patient_id="nonexistent_patient", procedure_type="routine_checkup"
            )

    # =========================================================================
    # Cost Calculation Tests
    # =========================================================================

    def test_calculate_cost_success(self):
        """Test calculating cost for a procedure."""
        result = self.tools.calculate_cost(
            appointment_type="routine_checkup",
            insurance_provider="BlueCross",
        )
        self.assertIn("base_cost", result)
        self.assertIn("copay", result)
        self.assertIn("insurance_covers", result)
        self.assertIn("patient_pays", result)
        self.assertIsInstance(result["base_cost"], (int, float))
        self.assertIsInstance(result["copay"], (int, float))
        self.assertIsInstance(result["insurance_covers"], (int, float))
        self.assertIsInstance(result["patient_pays"], (int, float))

    # =========================================================================
    # Prescription Tests
    # =========================================================================

    def test_get_prescription_details_success(self):
        """Test getting prescription details."""
        prescription_id = list(self.db.prescriptions.keys())[0]
        prescription = self.tools.get_prescription_details(prescription_id)
        self.assertIsNotNone(prescription)
        self.assertEqual(prescription.prescription_id, prescription_id)

    def test_get_prescription_details_not_found(self):
        """Test getting non-existent prescription."""
        with self.assertRaises(ValueError):
            self.tools.get_prescription_details("nonexistent_prescription")

    def test_request_prescription_refill_success(self):
        """Test requesting prescription refill."""
        prescription = next(
            (p for p in self.db.prescriptions.values() if p.refills_remaining > 0),
            None,
        )
        self.assertIsNotNone(prescription, "No prescription with refills available")

        initial_refills = prescription.refills_remaining
        result = self.tools.request_prescription_refill(
            patient_id=prescription.patient_id,
            prescription_id=prescription.prescription_id,
        )
        self.assertEqual(result.prescription_id, prescription.prescription_id)
        self.assertEqual(result.refills_remaining, initial_refills - 1)

    def test_request_prescription_refill_no_refills(self):
        """Test requesting refill when no refills remaining."""
        prescription = next(
            (p for p in self.db.prescriptions.values() if p.refills_remaining == 0),
            None,
        )
        self.assertIsNotNone(prescription, "No prescription with 0 refills available")

        with self.assertRaises(ValueError) as context:
            self.tools.request_prescription_refill(
                patient_id=prescription.patient_id,
                prescription_id=prescription.prescription_id,
            )
        self.assertIn("cannot refill prescription", str(context.exception).lower())

    def test_request_prescription_refill_not_found(self):
        """Test requesting refill for non-existent prescription."""
        with self.assertRaises(ValueError):
            self.tools.request_prescription_refill(
                patient_id="patient_001", prescription_id="nonexistent_prescription"
            )

    # =========================================================================
    # Test Results Tests
    # =========================================================================

    def test_check_test_results_success(self):
        """Test checking test results for a patient."""
        results = self.tools.check_test_results(patient_id="patient_001")
        self.assertIsInstance(results, list)

    def test_check_test_results_patient_not_found(self):
        """Test checking test results for non-existent patient."""
        results = self.tools.check_test_results(patient_id="nonexistent_patient")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    # =========================================================================
    # Chronic Condition Monitoring Tests
    # =========================================================================

    def test_get_vital_signs_history_success(self):
        """Test getting vital signs history for a patient."""
        result = self.tools.get_vital_signs_history(patient_id="patient_001")
        self.assertIsInstance(result, list)
        for record in result:
            self.assertIn("date", record)
            self.assertIn("vital_type", record)
            self.assertIn("value", record)

    def test_get_vital_signs_history_patient_not_found(self):
        """Test getting vital signs for non-existent patient."""
        with self.assertRaises(ValueError):
            self.tools.get_vital_signs_history(patient_id="nonexistent_patient")

    def test_get_chronic_conditions_success(self):
        """Test getting chronic conditions for a patient."""
        result = self.tools.get_chronic_conditions(patient_id="patient_001")
        self.assertIsInstance(result, list)
        for condition in result:
            self.assertIn("condition_name", condition)
            self.assertIn("diagnosed_date", condition)

    def test_get_chronic_conditions_patient_not_found(self):
        """Test getting chronic conditions for non-existent patient."""
        with self.assertRaises(ValueError):
            self.tools.get_chronic_conditions(patient_id="nonexistent_patient")

    def test_get_lab_results_success(self):
        """Test getting lab results for a patient."""
        result = self.tools.get_lab_results(patient_id="patient_001")
        self.assertIsInstance(result, list)

    def test_get_lab_results_patient_not_found(self):
        """Test getting lab results for non-existent patient."""
        with self.assertRaises(ValueError):
            self.tools.get_lab_results(patient_id="nonexistent_patient")

    # =========================================================================
    # Transfer/Escalation Tests
    # =========================================================================

    def test_transfer_to_nurse(self):
        """Test transferring to nurse."""
        result = self.tools.transfer_to_nurse()
        self.assertIsInstance(result, str)
        self.assertIn("transfer", result.lower())

    def test_transfer_to_human_agent(self):
        """Test transferring to human agent."""
        result = self.tools.transfer_to_human_agent()
        self.assertIsInstance(result, str)
        self.assertIn("transfer", result.lower())

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def assertHasAttr(self, obj, attr):
        """Assert that object has attribute."""
        self.assertTrue(hasattr(obj, attr), f"Object does not have attribute '{attr}'")


if __name__ == "__main__":
    unittest.main()
