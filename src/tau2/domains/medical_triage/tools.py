"""Toolkit for the medical triage system."""

from typing import Optional

from loguru import logger

from tau2.domains.medical_triage.data_model import (
    Appointment,
    MedicalDB,
    Patient,
    Provider,
    Referral,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class MedicalTriageTools(ToolKitBase):
    """All tools for the medical triage domain."""

    db: MedicalDB

    def __init__(self, db: MedicalDB) -> None:
        super().__init__(db)

    def _get_patient(self, patient_id: str) -> Patient:
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient {patient_id} not found")
        return self.db.patients[patient_id]

    def _get_provider(self, provider_id: str) -> Provider:
        if provider_id not in self.db.providers:
            raise ValueError(f"Provider {provider_id} not found")
        return self.db.providers[provider_id]

    def _get_appointment(self, appointment_id: str) -> Appointment:
        if appointment_id not in self.db.appointments:
            raise ValueError(f"Appointment {appointment_id} not found")
        return self.db.appointments[appointment_id]

    def _get_new_appointment_id(self) -> str:
        existing = set(self.db.appointments.keys())
        for i in range(1, 10000):
            aid = f"APT{i:05d}"
            if aid not in existing:
                return aid
        raise ValueError("No available appointment IDs")

    def _get_new_referral_id(self) -> str:
        existing = set(self.db.referrals.keys())
        for i in range(1, 10000):
            rid = f"REF{i:05d}"
            if rid not in existing:
                return rid
        raise ValueError("No available referral IDs")

    # ── Read Tools ──────────────────────────────────────────────────────────

    @is_tool(tool_type=ToolType.READ)
    def find_patient_by_name_dob(
        self, first_name: str, last_name: str, date_of_birth: str
    ) -> dict:
        """Find a patient by name and date of birth. Returns patient_id if found."""
        for pid, patient in self.db.patients.items():
            if (
                patient.first_name.lower() == first_name.lower()
                and patient.last_name.lower() == last_name.lower()
                and patient.date_of_birth == date_of_birth
            ):
                return {"patient_id": pid}
        raise ValueError("Patient not found with provided name and date of birth")

    @is_tool(tool_type=ToolType.READ)
    def get_patient_details(self, patient_id: str) -> dict:
        """Get full patient profile."""
        return self._get_patient(patient_id).model_dump()

    @is_tool(tool_type=ToolType.READ)
    def get_patient_allergies(self, patient_id: str) -> dict:
        """Get patient's known allergies."""
        patient = self._get_patient(patient_id)
        return {"patient_id": patient_id, "allergies": patient.allergies}

    @is_tool(tool_type=ToolType.READ)
    def get_patient_medications(self, patient_id: str) -> dict:
        """Get patient's current medications."""
        patient = self._get_patient(patient_id)
        return {"patient_id": patient_id, "medications": patient.current_medications}

    @is_tool(tool_type=ToolType.READ)
    def get_patient_medical_history(self, patient_id: str) -> dict:
        """Get patient's medical history."""
        patient = self._get_patient(patient_id)
        return {"patient_id": patient_id, "history": patient.medical_history}

    @is_tool(tool_type=ToolType.READ)
    def get_patient_insurance(self, patient_id: str) -> dict:
        """Get patient's insurance details including referral requirements."""
        patient = self._get_patient(patient_id)
        return {"patient_id": patient_id, "insurance": patient.insurance.model_dump()}

    @is_tool(tool_type=ToolType.READ)
    def get_appointment_details(self, appointment_id: str) -> dict:
        """Get details of a specific appointment."""
        return self._get_appointment(appointment_id).model_dump()

    @is_tool(tool_type=ToolType.READ)
    def get_patient_appointments(self, patient_id: str) -> dict:
        """Get all appointments for a patient."""
        patient = self._get_patient(patient_id)
        appointments = []
        for aid in patient.appointments:
            if aid in self.db.appointments:
                appointments.append(self.db.appointments[aid].model_dump())
        return {"patient_id": patient_id, "appointments": appointments}

    @is_tool(tool_type=ToolType.READ)
    def get_available_slots(
        self, department: str, start_date: str, end_date: Optional[str] = None
    ) -> dict:
        """Get available appointment slots for a department within a date range."""
        if end_date is None:
            end_date = start_date
        slots = []
        for pid, provider in self.db.providers.items():
            if provider.department == department or provider.specialty == department:
                for slot in provider.available_slots:
                    slot_date = slot[:10] if len(slot) >= 10 else slot
                    if start_date <= slot_date <= end_date:
                        slots.append(
                            {
                                "provider_id": pid,
                                "provider_name": provider.name,
                                "datetime": slot,
                                "clinic": provider.clinic,
                            }
                        )
        slots.sort(key=lambda s: s["datetime"])
        return {"department": department, "available_slots": slots}

    @is_tool(tool_type=ToolType.READ)
    def get_provider_details(self, provider_id: str) -> dict:
        """Get details of a healthcare provider."""
        return self._get_provider(provider_id).model_dump()

    @is_tool(tool_type=ToolType.READ)
    def check_referral_status(self, patient_id: str, department: str) -> dict:
        """Check if patient has an active referral for a specific department."""
        for rid, ref in self.db.referrals.items():
            if (
                ref.patient_id == patient_id
                and ref.target_department == department
                and ref.status == "active"
                and not ref.used
            ):
                return {
                    "has_referral": True,
                    "referral_id": rid,
                    "referral": ref.model_dump(),
                }
        return {"has_referral": False, "referral_id": None}

    # ── Write Tools ─────────────────────────────────────────────────────────

    @is_tool(tool_type=ToolType.WRITE)
    def schedule_appointment(
        self,
        patient_id: str,
        appointment_type: str,
        department: str,
        datetime_slot: str,
        provider_id: str,
        reason: str,
        triage_level: Optional[str] = None,
        referral_id: Optional[str] = None,
    ) -> dict:
        """Schedule a new appointment for a patient."""
        patient = self._get_patient(patient_id)
        provider = self._get_provider(provider_id)

        active_apts = [
            aid
            for aid in patient.appointments
            if aid in self.db.appointments
            and self.db.appointments[aid].status == "scheduled"
        ]
        if len(active_apts) >= 3:
            raise ValueError(
                "Patient already has 3 active appointments. Cancel one first."
            )

        if datetime_slot not in provider.available_slots:
            raise ValueError(
                f"Slot {datetime_slot} not available for provider {provider_id}"
            )

        if (
            appointment_type == "specialist"
            and patient.insurance.referral_required
            and department not in ("general_medicine", "primary_care")
        ):
            if referral_id is None:
                ref_check = self.check_referral_status(patient_id, department)
                if not ref_check["has_referral"]:
                    raise ValueError(
                        f"Patient's insurance requires a referral for {department}."
                    )
                referral_id = ref_check["referral_id"]

        apt_id = self._get_new_appointment_id()
        appointment = Appointment(
            appointment_id=apt_id,
            patient_id=patient_id,
            provider_id=provider_id,
            datetime=datetime_slot,
            type=appointment_type,
            reason=reason,
            status="scheduled",
            triage_level=triage_level,
            referral_id=referral_id,
        )
        self.db.appointments[apt_id] = appointment
        patient.appointments.append(apt_id)
        provider.available_slots.remove(datetime_slot)

        if referral_id and referral_id in self.db.referrals:
            self.db.referrals[referral_id].used = True

        logger.info(f"Appointment {apt_id} scheduled for patient {patient_id}")
        return {"appointment_id": apt_id, "status": "scheduled"}

    @is_tool(tool_type=ToolType.WRITE)
    def cancel_appointment(self, appointment_id: str) -> dict:
        """Cancel an existing appointment."""
        apt = self._get_appointment(appointment_id)
        if apt.status != "scheduled":
            raise ValueError(
                f"Appointment {appointment_id} is not scheduled (status: {apt.status})"
            )
        apt.status = "cancelled"
        if apt.provider_id in self.db.providers:
            self.db.providers[apt.provider_id].available_slots.append(apt.datetime)
        return {"appointment_id": appointment_id, "status": "cancelled"}

    @is_tool(tool_type=ToolType.WRITE)
    def reschedule_appointment(
        self,
        appointment_id: str,
        new_datetime: str,
        new_provider_id: Optional[str] = None,
    ) -> dict:
        """Reschedule an existing appointment."""
        apt = self._get_appointment(appointment_id)
        if apt.status != "scheduled":
            raise ValueError(f"Appointment {appointment_id} is not scheduled")
        if apt.reschedule_count >= 2:
            raise ValueError(
                "Appointment rescheduled 2 times already. Cancel and create new."
            )

        provider_id = new_provider_id or apt.provider_id
        provider = self._get_provider(provider_id)
        if new_datetime not in provider.available_slots:
            raise ValueError(f"Slot {new_datetime} not available")

        if apt.provider_id in self.db.providers:
            self.db.providers[apt.provider_id].available_slots.append(apt.datetime)

        apt.datetime = new_datetime
        apt.provider_id = provider_id
        apt.reschedule_count += 1
        provider.available_slots.remove(new_datetime)

        return {
            "appointment_id": appointment_id,
            "new_datetime": new_datetime,
            "reschedule_count": apt.reschedule_count,
        }

    @is_tool(tool_type=ToolType.WRITE)
    def create_referral(
        self, patient_id: str, referring_provider_id: str, department: str, reason: str
    ) -> dict:
        """Create a referral for a patient to a specialist department."""
        self._get_patient(patient_id)
        self._get_provider(referring_provider_id)
        ref_id = self._get_new_referral_id()
        referral = Referral(
            referral_id=ref_id,
            patient_id=patient_id,
            referring_provider_id=referring_provider_id,
            target_department=department,
            reason=reason,
            status="active",
            created_date="2026-03-15",
            expiry_date="2026-06-15",
        )
        self.db.referrals[ref_id] = referral
        return {"referral_id": ref_id, "status": "active"}

    @is_tool(tool_type=ToolType.WRITE)
    def add_triage_notes(
        self, appointment_id: str, notes: str, triage_level: str
    ) -> dict:
        """Add triage assessment notes and level to an appointment."""
        apt = self._get_appointment(appointment_id)
        apt.notes = notes
        apt.triage_level = triage_level
        return {
            "appointment_id": appointment_id,
            "triage_level": triage_level,
            "notes_added": True,
        }

    @is_tool(tool_type=ToolType.WRITE)
    def log_emergency_escalation(
        self, patient_id: str, symptoms: str, triage_level: str, action_taken: str
    ) -> dict:
        """Log an emergency escalation event."""
        patient = self._get_patient(patient_id)
        logger.warning(
            f"EMERGENCY: Patient {patient_id} ({patient.first_name} {patient.last_name}) — {triage_level} — {symptoms}"
        )
        return {
            "escalation_logged": True,
            "patient_id": patient_id,
            "triage_level": triage_level,
        }

    @is_tool(tool_type=ToolType.WRITE)
    def transfer_to_human_clinician(self, patient_id: str, reason: str) -> dict:
        """Transfer the patient to a human clinician."""
        patient = self._get_patient(patient_id)
        logger.info(
            f"Transfer to clinician: {patient_id} ({patient.first_name} {patient.last_name}) — {reason}"
        )
        return {"transferred": True, "patient_id": patient_id, "reason": reason}
