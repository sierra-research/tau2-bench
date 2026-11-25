from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.healthcare.utils import HEALTHCARE_DB_PATH
from tau2.environment.db import DB

# Type definitions
AppointmentType = Literal["routine_checkup", "follow_up", "urgent_care", "specialist"]
AppointmentStatus = Literal["scheduled", "completed", "cancelled", "no_show"]
InsuranceProvider = Literal[
    "BlueCross", "Aetna", "UnitedHealth", "Medicare", "Medicaid", "SelfPay"
]
PrescriptionStatus = Literal["active", "expired", "refill_needed", "discontinued"]
TestResultStatus = Literal["pending", "ready", "reviewed"]
ConditionSeverity = Literal["mild", "moderate", "severe"]
MedicationRoute = Literal["oral", "injection", "topical", "inhaled"]
AllergySeverity = Literal["mild", "moderate", "severe", "life_threatening"]
LabResultStatus = Literal["pending", "resulted", "reviewed"]
Priority = Literal["routine", "urgent", "stat"]


class Name(BaseModel):
    """Patient or doctor name."""

    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")


class InsurancePlan(BaseModel):
    """Insurance plan information."""

    provider: InsuranceProvider = Field(description="Insurance provider name")
    policy_number: str = Field(description="Insurance policy number")
    group_number: str = Field(description="Insurance group number")
    copay_amount: int = Field(description="Standard copay amount in dollars")
    coverage_details: str = Field(description="Brief description of coverage")


class ContactInfo(BaseModel):
    """Contact information."""

    phone: str = Field(description="Phone number")
    email: str = Field(description="Email address")
    address: str = Field(description="Full address")


class MedicalCondition(BaseModel):
    """Chronic medical condition with diagnostic information."""

    condition_name: str = Field(description="Name of the medical condition")
    icd10_code: str = Field(description="ICD-10 diagnostic code")
    diagnosed_date: str = Field(description="Date diagnosed in YYYY-MM-DD format")
    severity: ConditionSeverity = Field(description="Severity of the condition")
    controlled: bool = Field(description="Whether the condition is well-controlled")
    requires_monitoring: bool = Field(
        description="Whether regular monitoring is required"
    )


class Medication(BaseModel):
    """Current medication with detailed information."""

    medication_id: str = Field(description="Unique medication identifier")
    name: str = Field(description="Brand or generic name of medication")
    generic_name: str = Field(description="Generic pharmaceutical name")
    dosage: str = Field(description="Dosage amount (e.g., '10mg', '500mg')")
    frequency: str = Field(
        description="How often taken (e.g., 'once daily', 'twice daily')"
    )
    route: MedicationRoute = Field(description="Route of administration")
    prescribed_date: str = Field(description="Date prescribed in YYYY-MM-DD format")
    indication: str = Field(description="Medical reason for prescription")
    interactions: List[str] = Field(
        default_factory=list, description="List of medications this interacts with"
    )
    side_effects: List[str] = Field(
        default_factory=list, description="Common side effects"
    )


class Allergy(BaseModel):
    """Allergy with reaction details and severity."""

    allergen: str = Field(description="Substance causing allergic reaction")
    reaction_type: str = Field(
        description="Type of reaction (e.g., 'rash', 'anaphylaxis', 'hives')"
    )
    severity: AllergySeverity = Field(description="Severity of allergic reaction")
    onset_date: Optional[str] = Field(
        default=None, description="When allergy was first identified"
    )


class VitalSigns(BaseModel):
    """Vital signs measurement with timestamp."""

    timestamp: str = Field(description="When vitals were measured (ISO format)")
    blood_pressure_systolic: Optional[int] = Field(
        default=None, description="Systolic BP in mmHg"
    )
    blood_pressure_diastolic: Optional[int] = Field(
        default=None, description="Diastolic BP in mmHg"
    )
    heart_rate: Optional[int] = Field(
        default=None, description="Heart rate in beats per minute"
    )
    temperature: Optional[float] = Field(
        default=None, description="Body temperature in °F"
    )
    respiratory_rate: Optional[int] = Field(
        default=None, description="Breaths per minute"
    )
    oxygen_saturation: Optional[int] = Field(
        default=None, description="SpO2 percentage"
    )
    weight: Optional[float] = Field(default=None, description="Weight in kg")
    height: Optional[float] = Field(default=None, description="Height in cm")


class LabResult(BaseModel):
    """Laboratory test result with detailed values."""

    test_id: str = Field(description="Unique test identifier")
    patient_id: str = Field(description="Patient ID")
    test_type: str = Field(
        description="Type of test (e.g., 'CBC', 'HbA1c', 'Lipid Panel')"
    )
    test_date: str = Field(description="Date test was performed in YYYY-MM-DD format")
    results: Dict[str, Dict[str, Union[float, int, str]]] = Field(
        description="Test results with values, units, and flags"
    )
    ordering_doctor: str = Field(description="Doctor who ordered the test")
    status: LabResultStatus = Field(description="Status of the lab result")
    critical: bool = Field(
        default=False,
        description="Whether result is critical and requires immediate action",
    )


class LabOrder(BaseModel):
    """Laboratory test order."""

    order_id: str = Field(description="Unique order identifier")
    patient_id: str = Field(description="Patient ID")
    test_type: str = Field(description="Type of test ordered")
    priority: Priority = Field(description="Priority level")
    clinical_indication: str = Field(description="Medical reason for ordering test")
    ordered_date: str = Field(description="Date ordered in YYYY-MM-DD format")
    status: Literal["pending", "completed", "cancelled"] = Field(
        description="Status of the order"
    )


class EmergencyTransfer(BaseModel):
    """Emergency transfer record."""

    patient_id: str = Field(description="Patient ID")
    reason: str = Field(description="Reason for emergency transfer")
    symptoms: List[str] = Field(description="List of concerning symptoms")
    timestamp: str = Field(description="When transfer was initiated (ISO format)")
    vital_signs: Optional[VitalSigns] = Field(
        default=None, description="Latest vital signs"
    )
    current_medications: List[str] = Field(
        default_factory=list, description="Medications patient is currently taking"
    )


class Patient(BaseModel):
    """Patient information in the system with comprehensive medical record."""

    patient_id: str = Field(description="Unique patient identifier")
    name: Name = Field(description="Patient name")
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")
    contact: ContactInfo = Field(description="Contact information")
    insurance: InsurancePlan = Field(description="Insurance plan")

    # Enriched medical data
    chronic_conditions: List[MedicalCondition] = Field(
        default_factory=list, description="Detailed chronic medical conditions"
    )
    current_medications_detailed: List[Medication] = Field(
        default_factory=list,
        description="Detailed current medications with interactions and side effects",
    )
    allergies_detailed: List[Allergy] = Field(
        default_factory=list,
        description="Detailed allergies with severity and reaction type",
    )
    vital_signs_history: List[VitalSigns] = Field(
        default_factory=list, description="Historical vital signs measurements"
    )

    # References to other records
    appointment_ids: List[str] = Field(
        default_factory=list, description="List of appointment IDs for this patient"
    )
    prescription_ids: List[str] = Field(
        default_factory=list, description="List of prescription IDs for this patient"
    )
    lab_result_ids: List[str] = Field(
        default_factory=list, description="List of lab result IDs for this patient"
    )

    # Clinical tracking
    last_consultation_date: Optional[str] = Field(
        default=None, description="Date of last consultation"
    )
    last_hba1c_date: Optional[str] = Field(
        default=None, description="Date of last HbA1c test (for diabetics)"
    )
    last_lipid_panel_date: Optional[str] = Field(
        default=None,
        description="Date of last lipid panel (for cardiovascular monitoring)",
    )

    # Risk flags
    high_risk_conditions: List[str] = Field(
        default_factory=list,
        description="List of high-risk conditions requiring special attention",
    )
    needs_urgent_follow_up: bool = Field(
        default=False, description="Whether patient requires urgent follow-up"
    )


class Doctor(BaseModel):
    """Doctor information in the system."""

    doctor_id: str = Field(description="Unique doctor identifier")
    name: Name = Field(description="Doctor name")
    specialty: str = Field(description="Medical specialty")
    available_days: List[str] = Field(
        description="Days of the week available (e.g., ['Monday', 'Wednesday', 'Friday'])"
    )
    available_times: List[str] = Field(
        description="Available time slots (e.g., ['09:00', '10:00', '14:00'])"
    )


class Appointment(BaseModel):
    """Appointment information."""

    appointment_id: str = Field(description="Unique appointment identifier")
    patient_id: str = Field(description="Patient ID")
    doctor_id: str = Field(description="Doctor ID")
    appointment_type: AppointmentType = Field(description="Type of appointment")
    date: str = Field(description="Appointment date in YYYY-MM-DD format")
    time: str = Field(description="Appointment time in HH:MM format (24-hour)")
    status: AppointmentStatus = Field(description="Current status of the appointment")
    reason: str = Field(description="Reason for visit")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    created_at: str = Field(description="When appointment was created (ISO format)")
    cost: int = Field(description="Appointment cost in dollars (before insurance)")


class Prescription(BaseModel):
    """Prescription information."""

    prescription_id: str = Field(description="Unique prescription identifier")
    patient_id: str = Field(description="Patient ID")
    doctor_id: str = Field(description="Prescribing doctor ID")
    medication_name: str = Field(description="Name of the medication")
    dosage: str = Field(description="Dosage instructions (e.g., '10mg once daily')")
    quantity: int = Field(description="Quantity prescribed")
    refills_remaining: int = Field(description="Number of refills remaining")
    status: PrescriptionStatus = Field(description="Current prescription status")
    prescribed_date: str = Field(description="Date prescribed in YYYY-MM-DD format")
    expiration_date: str = Field(description="Expiration date in YYYY-MM-DD format")


class TestResult(BaseModel):
    """Medical test result information."""

    test_id: str = Field(description="Unique test identifier")
    patient_id: str = Field(description="Patient ID")
    test_name: str = Field(description="Name of the test")
    test_date: str = Field(description="Date test was performed in YYYY-MM-DD format")
    status: TestResultStatus = Field(description="Status of the test result")
    result: Optional[str] = Field(
        default=None, description="Test result details (only if ready)"
    )
    notes: Optional[str] = Field(
        default=None, description="Doctor's notes on the result"
    )


class Payment(BaseModel):
    """Payment transaction."""

    payment_id: str = Field(description="Unique payment identifier")
    patient_id: str = Field(description="Patient ID")
    amount: int = Field(description="Payment amount in dollars")
    payment_method: Literal["credit_card", "debit_card", "insurance", "cash"] = Field(
        description="Method of payment"
    )
    date: str = Field(description="Payment date in YYYY-MM-DD format")
    description: str = Field(description="What the payment was for")


class HealthcareDB(DB):
    """
    Main database for the healthcare domain.
    Contains all patients, doctors, appointments, prescriptions, test results, and lab data.
    """

    patients: Dict[str, Patient] = Field(
        default_factory=dict, description="Dictionary of patients keyed by patient_id"
    )
    doctors: Dict[str, Doctor] = Field(
        default_factory=dict, description="Dictionary of doctors keyed by doctor_id"
    )
    appointments: Dict[str, Appointment] = Field(
        default_factory=dict,
        description="Dictionary of appointments keyed by appointment_id",
    )
    prescriptions: Dict[str, Prescription] = Field(
        default_factory=dict,
        description="Dictionary of prescriptions keyed by prescription_id",
    )
    test_results: Dict[str, TestResult] = Field(
        default_factory=dict, description="Dictionary of test results keyed by test_id"
    )
    payments: Dict[str, Payment] = Field(
        default_factory=dict, description="Dictionary of payments keyed by payment_id"
    )
    # New medical data collections
    lab_results: Dict[str, LabResult] = Field(
        default_factory=dict, description="Dictionary of lab results keyed by test_id"
    )
    lab_orders: Dict[str, LabOrder] = Field(
        default_factory=dict, description="Dictionary of lab orders keyed by order_id"
    )
    emergency_transfers: Dict[str, EmergencyTransfer] = Field(
        default_factory=dict,
        description="Dictionary of emergency transfers keyed by patient_id",
    )

    tool_call_history: List[str] = Field(
        default_factory=list,
        description="History of assistant tool calls made during the conversation (tool names only)",
    )

    @classmethod
    def get_db_path(cls):
        """Get the default database path."""
        return HEALTHCARE_DB_PATH


# Export types for convenience
__all__ = [
    # Main models
    "Patient",
    "Doctor",
    "Appointment",
    "Prescription",
    "TestResult",
    "Payment",
    "HealthcareDB",
    # New medical models
    "MedicalCondition",
    "Medication",
    "Allergy",
    "VitalSigns",
    "LabResult",
    "LabOrder",
    "EmergencyTransfer",
    # Type definitions
    "AppointmentType",
    "AppointmentStatus",
    "InsuranceProvider",
    "PrescriptionStatus",
    "TestResultStatus",
    "ConditionSeverity",
    "MedicationRoute",
    "AllergySeverity",
    "LabResultStatus",
    "Priority",
    # Supporting classes
    "Name",
    "InsurancePlan",
    "ContactInfo",
]
