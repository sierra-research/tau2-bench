from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.domains.healthcare.data_model import (
    Appointment,
    AppointmentStatus,
    AppointmentType,
    Doctor,
    HealthcareDB,
    Patient,
    Payment,
    Prescription,
    PrescriptionStatus,
    TestResult,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HealthcareTools(ToolKitBase):
    """All the tools for the healthcare domain (agent-side)."""

    db: HealthcareDB

    def __init__(self, db: HealthcareDB) -> None:
        super().__init__(db)

    # Helper methods

    def _get_patient(self, patient_id: str) -> Patient:
        """Get patient from database."""
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient {patient_id} not found")
        return self.db.patients[patient_id]

    def _find_patient_by_identity(
        self, full_name: str, date_of_birth: str
    ) -> Optional[Patient]:
        """Find patient by full name and date of birth."""
        for patient in self.db.patients.values():
            patient_full_name = f"{patient.name.first_name} {patient.name.last_name}"
            if (
                patient_full_name == full_name
                and patient.date_of_birth == date_of_birth
            ):
                return patient
        return None

    def _get_doctor(self, doctor_id: str) -> Doctor:
        """Get doctor from database."""
        if doctor_id not in self.db.doctors:
            raise ValueError(f"Doctor {doctor_id} not found")
        return self.db.doctors[doctor_id]

    def _get_appointment(self, appointment_id: str) -> Appointment:
        """Get appointment from database."""
        if appointment_id not in self.db.appointments:
            raise ValueError(f"Appointment {appointment_id} not found")
        return self.db.appointments[appointment_id]

    def _get_prescription(self, prescription_id: str) -> Prescription:
        """Get prescription from database."""
        if prescription_id not in self.db.prescriptions:
            raise ValueError(f"Prescription {prescription_id} not found")
        return self.db.prescriptions[prescription_id]

    def _get_new_appointment_id(self) -> str:
        """Generate a new appointment ID."""
        for i in range(1, 11):
            apt_id = f"APPT_NEW_{i:03d}"
            if apt_id not in self.db.appointments:
                return apt_id
        raise ValueError("Too many appointments created")

    def _get_new_payment_id(self) -> str:
        """Generate a new payment ID."""
        for i in range(1, 11):
            pay_id = f"PAY_NEW_{i:03d}"
            if pay_id not in self.db.payments:
                return pay_id
        raise ValueError("Too many payments created")

    def _get_current_datetime(self) -> str:
        """Get current datetime (fixed for simulation)."""
        return "2024-05-15T15:00:00"

    def _is_time_slot_available(self, doctor: Doctor, date: str, time: str) -> bool:
        """Check if a doctor has a specific time slot available."""
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        day_name = date_obj.strftime("%A")

        if day_name not in doctor.available_days:
            return False

        if time not in doctor.available_times:
            return False

        for apt in self.db.appointments.values():
            if (
                apt.doctor_id == doctor.doctor_id
                and apt.date == date
                and apt.time == time
                and apt.status == "scheduled"
            ):
                return False

        return True

    @is_tool(ToolType.READ)
    def get_patient_details(self, full_name: str, date_of_birth: str) -> Patient:
        """
        Retrieve complete patient information.

        Args:
            full_name: Patient's full name (e.g., "Sarah Johnson")
            date_of_birth: Patient's date of birth in YYYY-MM-DD format (e.g., "1985-03-15")

        Returns:
            Complete patient record with all details
        """
        patient = self._find_patient_by_identity(full_name, date_of_birth)
        if patient is None:
            raise ValueError(
                f"No patient found with name '{full_name}' and date of birth '{date_of_birth}'. Please verify the patient's identity information."
            )
        logger.info(
            f"Retrieved patient details for {full_name} (patient_id: {patient.patient_id})"
        )
        return patient

    @is_tool(ToolType.READ)
    def get_appointment_details(self, appointment_id: str) -> Appointment:
        """
        Retrieve details of a specific appointment.

        Args:
            appointment_id: The unique identifier for the appointment

        Returns:
            Complete appointment information
        """
        return self._get_appointment(appointment_id)

    @is_tool(ToolType.READ)
    def search_appointments(
        self,
        patient_id: str,
        status: Optional[AppointmentStatus] = None,
    ) -> List[Appointment]:
        """
        Search for appointments for a specific patient, optionally filtered by status.

        Args:
            patient_id: The patient ID to search appointments for
            status: Optional status filter (scheduled, completed, cancelled, no_show)

        Returns:
            List of matching appointments
        """
        patient = self._get_patient(patient_id)
        results = []

        for apt_id in patient.appointment_ids:
            if apt_id in self.db.appointments:
                apt = self.db.appointments[apt_id]
                if status is None or apt.status == status:
                    results.append(apt)

        return results

    @is_tool(ToolType.READ)
    def list_available_doctors(
        self,
        specialty: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[Doctor]:
        """
        List all doctors, optionally filtered by specialty and availability on a specific date.

        Args:
            specialty: Optional specialty to filter by (e.g., "General Practice", "Cardiology")
            date: Optional date in YYYY-MM-DD format to check availability

        Returns:
            List of doctors matching the criteria
        """
        results = []

        for doctor in self.db.doctors.values():
            if specialty and doctor.specialty.lower() != specialty.lower():
                continue

            if date:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                day_name = date_obj.strftime("%A")
                if day_name not in doctor.available_days:
                    continue

            results.append(doctor)

        return results

    @is_tool(ToolType.READ)
    def check_available_time_slots(
        self,
        doctor_id: str,
        date: str,
    ) -> List[str]:
        """
        Check what time slots are available for a specific doctor on a specific date.

        Args:
            doctor_id: The doctor's unique identifier
            date: Date to check in YYYY-MM-DD format

        Returns:
            List of available time slots in HH:MM format
        """
        doctor = self._get_doctor(doctor_id)
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        day_name = date_obj.strftime("%A")

        if day_name not in doctor.available_days:
            return []

        available_slots = []
        for time_slot in doctor.available_times:
            if self._is_time_slot_available(doctor, date, time_slot):
                available_slots.append(time_slot)

        return available_slots

    @is_tool(ToolType.WRITE)
    def book_appointment(
        self,
        patient_id: str,
        doctor_id: str,
        appointment_type: AppointmentType,
        date: str,
        time: str,
        reason: str,
    ) -> Appointment:
        """
        Book a new appointment for a patient.

        Args:
            patient_id: The patient's unique identifier
            doctor_id: The doctor's unique identifier
            appointment_type: Type of appointment (routine_checkup, follow_up, urgent_care, specialist)
            date: Appointment date in YYYY-MM-DD format
            time: Appointment time in HH:MM format (24-hour)
            reason: Reason for the visit

        Returns:
            The newly created appointment
        """
        patient = self._get_patient(patient_id)
        doctor = self._get_doctor(doctor_id)

        if not self._is_time_slot_available(doctor, date, time):
            raise ValueError(
                f"Time slot {time} on {date} is not available for Dr. {doctor.name.last_name}"
            )

        base_costs = {
            "routine_checkup": 150,
            "follow_up": 100,
            "urgent_care": 200,
            "specialist": 250,
        }
        cost = base_costs.get(appointment_type, 150)

        apt_id = self._get_new_appointment_id()
        appointment = Appointment(
            appointment_id=apt_id,
            patient_id=patient_id,
            doctor_id=doctor_id,
            appointment_type=appointment_type,
            date=date,
            time=time,
            status="scheduled",
            reason=reason,
            notes=None,
            created_at=self._get_current_datetime(),
            cost=cost,
        )

        self.db.appointments[apt_id] = appointment
        patient.appointment_ids.append(apt_id)

        logger.info(f"Booked appointment {apt_id} for patient {patient_id}")
        return appointment

    @is_tool(ToolType.WRITE)
    def cancel_appointment(
        self,
        appointment_id: str,
        reason: str,
    ) -> Appointment:
        """
        Cancel an existing appointment.

        Args:
            appointment_id: The appointment ID to cancel
            reason: Reason for cancellation

        Returns:
            The updated appointment with cancelled status
        """
        appointment = self._get_appointment(appointment_id)

        if appointment.status in ["cancelled", "completed"]:
            raise ValueError(
                f"Cannot cancel appointment with status: {appointment.status}"
            )

        appointment.status = "cancelled"
        if appointment.notes:
            appointment.notes += f" | Cancelled: {reason}"
        else:
            appointment.notes = f"Cancelled: {reason}"

        logger.info(f"Cancelled appointment {appointment_id}")
        return appointment

    @is_tool(ToolType.WRITE)
    def reschedule_appointment(
        self,
        appointment_id: str,
        new_date: str,
        new_time: str,
    ) -> Appointment:
        """
        Reschedule an existing appointment to a new date and time.

        Args:
            appointment_id: The appointment ID to reschedule
            new_date: New date in YYYY-MM-DD format
            new_time: New time in HH:MM format

        Returns:
            The updated appointment
        """
        appointment = self._get_appointment(appointment_id)

        if appointment.status != "scheduled":
            raise ValueError(
                f"Cannot reschedule appointment with status: {appointment.status}"
            )

        doctor = self._get_doctor(appointment.doctor_id)
        if not self._is_time_slot_available(doctor, new_date, new_time):
            raise ValueError(f"Time slot {new_time} on {new_date} is not available")

        old_date = appointment.date
        old_time = appointment.time
        appointment.date = new_date
        appointment.time = new_time
        if appointment.notes:
            appointment.notes += f" | Rescheduled from {old_date} {old_time}"
        else:
            appointment.notes = f"Rescheduled from {old_date} {old_time}"

        logger.info(
            f"Rescheduled appointment {appointment_id} to {new_date} {new_time}"
        )
        return appointment

    @is_tool(ToolType.READ)
    def verify_insurance_coverage(
        self,
        patient_id: str,
        procedure_type: Optional[str] = None,
    ) -> dict:
        """
        Verify patient's insurance coverage and copay information.

        Args:
            patient_id: The patient's unique identifier
            procedure_type: Optional specific procedure to check coverage for

        Returns:
            Dictionary with insurance verification details
        """
        patient = self._get_patient(patient_id)
        insurance = patient.insurance

        result = {
            "verified": True,
            "provider": insurance.provider,
            "policy_number": insurance.policy_number,
            "copay_amount": insurance.copay_amount,
            "coverage_details": insurance.coverage_details,
        }

        if procedure_type:
            result["procedure_covered"] = (
                "routine" in insurance.coverage_details.lower()
            )

        return result

    @is_tool(ToolType.READ)
    def get_prescription_details(self, prescription_id: str) -> Prescription:
        """
        Get details of a specific prescription.

        Args:
            prescription_id: The prescription's unique identifier

        Returns:
            Complete prescription information
        """
        return self._get_prescription(prescription_id)

    @is_tool(ToolType.WRITE)
    def request_prescription_refill(
        self,
        prescription_id: str,
        patient_id: str,
    ) -> Prescription:
        """
        Request a refill for an existing prescription. Checks if refills are available.

        Args:
            prescription_id: The prescription ID to refill
            patient_id: The patient's ID (for verification)

        Returns:
            Updated prescription information
        """
        prescription = self._get_prescription(prescription_id)

        if prescription.patient_id != patient_id:
            raise ValueError(
                f"Prescription {prescription_id} does not belong to patient {patient_id}"
            )

        if prescription.status != "active":
            raise ValueError(
                f"Cannot refill prescription with status: {prescription.status}"
            )

        if prescription.refills_remaining <= 0:
            raise ValueError(
                "No refills remaining. Patient needs to contact doctor for new prescription."
            )

        prescription.refills_remaining -= 1

        if prescription.refills_remaining == 0:
            prescription.status = "refill_needed"

        logger.info(f"Processed refill for prescription {prescription_id}")
        return prescription

    @is_tool(ToolType.READ)
    def check_test_results(
        self,
        patient_id: str,
        test_id: Optional[str] = None,
    ) -> List[TestResult]:
        """
        Check test results for a patient. If test_id provided, returns that specific test.

        Args:
            patient_id: The patient's unique identifier
            test_id: Optional specific test ID to retrieve

        Returns:
            List of test results (or single test if test_id provided)
        """
        if test_id:
            test = self.db.test_results.get(test_id)
            if not test:
                raise ValueError(f"Test {test_id} not found")
            if test.patient_id != patient_id:
                raise ValueError(
                    f"Test {test_id} does not belong to patient {patient_id}"
                )
            return [test]

        results = []
        for test in self.db.test_results.values():
            if test.patient_id == patient_id:
                results.append(test)

        return results

    @is_tool(ToolType.GENERIC)
    def calculate_cost(
        self,
        appointment_type: AppointmentType,
        insurance_provider: str,
    ) -> dict:
        """
        Calculate the estimated cost for an appointment including insurance copay.

        Args:
            appointment_type: Type of appointment
            insurance_provider: Patient's insurance provider

        Returns:
            Dictionary with cost breakdown
        """
        base_costs = {
            "routine_checkup": 150,
            "follow_up": 100,
            "urgent_care": 200,
            "specialist": 250,
        }

        copay_amounts = {
            "BlueCross": 20,
            "Aetna": 25,
            "UnitedHealth": 20,
            "Medicare": 0,
            "Medicaid": 0,
            "SelfPay": 0,
        }

        base_cost = base_costs.get(appointment_type, 150)
        copay = copay_amounts.get(insurance_provider, 30)

        return {
            "base_cost": base_cost,
            "copay": copay if insurance_provider != "SelfPay" else base_cost,
            "insurance_covers": base_cost - copay
            if insurance_provider != "SelfPay"
            else 0,
            "patient_pays": copay if insurance_provider != "SelfPay" else base_cost,
        }

    @is_tool(ToolType.GENERIC)
    def transfer_to_nurse(self) -> str:
        """
        Transfer the patient to a nurse for clinical questions or triage.

        Returns:
            Transfer confirmation message
        """
        logger.info("Transferring patient to nurse")
        return "Transferring you to a nurse who can better assist with your clinical questions. Please hold."

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agent(self) -> str:
        """
        Transfer the patient to a human agent when the request cannot be handled automatically.

        Returns:
            Transfer confirmation message
        """
        logger.info("Transferring patient to human agent")
        return "I'm transferring you to a specialist who can better assist you. Please hold."

    def create_appointment_for_test(
        self,
        appointment_id: str,
        patient_id: str,
        doctor_id: str,
        appointment_type: str,
        date: str,
        time: str,
        status: str,
        reason: str,
        cost: int,
        notes: Optional[str] = None,
    ) -> None:
        """Create an appointment for test scenario initialization."""
        from datetime import datetime
        from tau2.domains.healthcare.data_model import Appointment

        appointment = Appointment(
            appointment_id=appointment_id,
            patient_id=patient_id,
            doctor_id=doctor_id,
            appointment_type=appointment_type,  # type: ignore
            date=date,
            time=time,
            status=status,  # type: ignore
            reason=reason,
            notes=notes,
            created_at=datetime.now().isoformat(),
            cost=cost,
        )

        self.db.appointments[appointment_id] = appointment

        patient = self.db.patients.get(patient_id)
        if patient:
            if patient.appointment_ids is None:
                patient.appointment_ids = []
            patient.appointment_ids.append(appointment_id)

        logger.info(
            f"Created test appointment: {appointment_id} for patient {patient_id}"
        )

    def assert_appointment_exists(
        self, patient_id: str, appointment_type: Optional[str] = None
    ) -> bool:
        """Assert that an appointment exists for the patient."""
        patient = self.db.patients.get(patient_id)
        if not patient:
            return False

        if not patient.appointment_ids:
            return False

        if appointment_type is None:
            return True

        for apt_id in patient.appointment_ids:
            apt = self.db.appointments.get(apt_id)
            if apt and apt.appointment_type == appointment_type:
                return True

        return False

    def assert_appointment_status(
        self, appointment_id: str, expected_status: str
    ) -> bool:
        """Assert an appointment's status (scheduled, cancelled, completed)."""
        apt = self.db.appointments.get(appointment_id)
        if not apt:
            return False
        return apt.status == expected_status

    def assert_prescription_refills_remaining(
        self, prescription_id: str, expected_count: int
    ) -> bool:
        """Assert the number of refills remaining on a prescription."""
        rx = self.db.prescriptions.get(prescription_id)
        if not rx:
            return False
        return rx.refills_remaining == expected_count

    def assert_prescription_status(
        self, prescription_id: str, expected_status: str
    ) -> bool:
        """Assert a prescription's status (active, expired, discontinued)."""
        rx = self.db.prescriptions.get(prescription_id)
        if not rx:
            return False
        return rx.status == expected_status

    def assert_patient_has_insurance(self, patient_id: str, expected: bool) -> bool:
        """Assert whether a patient has insurance information on file."""
        patient = self.db.patients.get(patient_id)
        if not patient:
            return False
        has_insurance = patient.insurance is not None
        return has_insurance == expected

    def assert_insurance_provider(
        self, patient_id: str, expected_provider: str
    ) -> bool:
        """Assert the patient's insurance provider."""
        patient = self.db.patients.get(patient_id)
        if not patient or not patient.insurance:
            return False
        return patient.insurance.provider == expected_provider

    def assert_appointment_count_exceeds_baseline(self) -> bool:
        """Assert that real appointments exceed baseline + markers."""
        num_markers = sum(
            1
            for apt in self.db.appointments.values()
            if apt.date == "2024-01-01" and apt.time == "00:00"
        )
        num_real = len(self.db.appointments) - num_markers

        if num_markers == 0:
            return True

        return num_real >= (2 + num_markers)

    def assert_tool_was_called(self, tool_name: str) -> bool:
        """
        Verify that a specific tool was called during the conversation.

        Args:
            tool_name: Name of the tool to check (e.g., "get_prescription_details")

        Returns:
            True if the tool was called at least once, False otherwise
        """
        return tool_name in self.db.tool_call_history

    def assert_tool_was_not_called(self, tool_name: str) -> bool:
        """
        Verify that a specific tool was NOT called during the conversation.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool was never called, False if it was called
        """
        return tool_name not in self.db.tool_call_history

    def set_prescription_refills(self, prescription_id: str, refills: int) -> None:
        """Set the number of refills remaining on a prescription."""
        if prescription_id not in self.db.prescriptions:
            raise ValueError(f"Prescription {prescription_id} not found")
        self.db.prescriptions[prescription_id].refills_remaining = refills

    def set_prescription_status(self, prescription_id: str, status: str) -> None:
        """Set the status of a prescription (active, expired, discontinued, refill_needed)."""
        if prescription_id not in self.db.prescriptions:
            raise ValueError(f"Prescription {prescription_id} not found")
        self.db.prescriptions[prescription_id].status = status

    def set_prescription_medication(
        self, prescription_id: str, medication_name: str, dosage: str
    ) -> None:
        """Set the medication name and dosage for a prescription."""
        if prescription_id not in self.db.prescriptions:
            raise ValueError(f"Prescription {prescription_id} not found")
        self.db.prescriptions[prescription_id].medication_name = medication_name
        self.db.prescriptions[prescription_id].dosage = dosage

    def create_appointment_marker(
        self,
        appointment_id: str,
        patient_id: str,
        reason: str = "Pending booking request marker",
    ) -> None:
        """Create a temporary appointment marker to indicate pending booking request."""
        from tau2.domains.healthcare.data_model import Appointment
        from datetime import datetime

        marker_appt = Appointment(
            appointment_id=appointment_id,
            patient_id=patient_id,
            doctor_id="doc_001",
            appointment_type="routine_checkup",
            date="2024-01-01",  # Placeholder date
            time="00:00",
            status="scheduled",
            reason=reason,
            created_at=datetime.now().isoformat(),
            cost=0,
        )
        self.db.appointments[appointment_id] = marker_appt

    @is_tool(ToolType.READ)
    def get_vital_signs_history(
        self, patient_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Retrieve vital signs history for the specified number of days.

        Args:
            patient_id: Patient identifier
            days: Number of days to look back (default 30)

        Returns:
            List of vital signs measurements with timestamps
        """
        from datetime import datetime, timedelta

        patient = self._get_patient(patient_id)
        cutoff = datetime.now() - timedelta(days=days)

        recent_vitals = [
            vs
            for vs in patient.vital_signs_history
            if datetime.fromisoformat(vs.timestamp) > cutoff
        ]

        result = []
        for vs in recent_vitals:
            result.append(
                {
                    "timestamp": vs.timestamp,
                    "blood_pressure": f"{vs.blood_pressure_systolic}/{vs.blood_pressure_diastolic}"
                    if vs.blood_pressure_systolic
                    else None,
                    "heart_rate": vs.heart_rate,
                    "temperature": vs.temperature,
                    "respiratory_rate": vs.respiratory_rate,
                    "oxygen_saturation": vs.oxygen_saturation,
                    "weight": vs.weight,
                    "height": vs.height,
                }
            )

        logger.info(f"Retrieved {len(result)} vital signs for patient {patient_id}")
        return result

    @is_tool(ToolType.READ)
    def get_lab_results(
        self, patient_id: str, test_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve laboratory test results for a patient.

        Args:
            patient_id: Patient identifier
            test_type: Optional filter for specific test type (e.g., 'HbA1c', 'Lipid Panel')

        Returns:
            List of lab results sorted by date (most recent first)
        """
        patient = self._get_patient(patient_id)

        results = [
            self.db.lab_results[lab_id]
            for lab_id in patient.lab_result_ids
            if lab_id in self.db.lab_results
        ]

        if test_type:
            results = [r for r in results if r.test_type == test_type]

        results = sorted(results, key=lambda x: x.test_date, reverse=True)

        result = []
        for lab in results:
            result.append(
                {
                    "test_id": lab.test_id,
                    "test_type": lab.test_type,
                    "test_date": lab.test_date,
                    "results": lab.results,
                    "status": lab.status,
                    "critical": lab.critical,
                    "ordering_doctor": lab.ordering_doctor,
                }
            )

        logger.info(f"Retrieved {len(result)} lab results for patient {patient_id}")
        return result

    @is_tool(ToolType.READ)
    def get_chronic_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get detailed information about patient's chronic medical conditions.

        Args:
            patient_id: Patient identifier

        Returns:
            List of chronic conditions with severity and control status
        """
        patient = self._get_patient(patient_id)

        conditions = []
        for condition in patient.chronic_conditions:
            conditions.append(
                {
                    "condition_name": condition.condition_name,
                    "icd10_code": condition.icd10_code,
                    "diagnosed_date": condition.diagnosed_date,
                    "severity": condition.severity,
                    "controlled": condition.controlled,
                    "requires_monitoring": condition.requires_monitoring,
                }
            )

        logger.info(
            f"Retrieved {len(conditions)} chronic conditions for patient {patient_id}"
        )
        return conditions
