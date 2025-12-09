from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


# Common literals
YesNo = Literal["yes", "no"]
ControlledSubstanceSchedule = Literal["II", "III", "IV", "V", "none"]
PrescriptionStatus = Literal[
    "active", "filled", "on-hold", "expired", "canceled", "transferred"
]


# Shared simple models
class Name(BaseModel):
    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


# Medication models
class MedicationImage(BaseModel):
    url: str = Field(description="URL to the image")
    view: Literal["front", "back"] = Field(description="Image view angle")


class MedicationStorage(BaseModel):
    temperature_c: str = Field(
        description='Recommended storage temperature in Celsius (e.g., "20-25")'
    )
    protect_from_light: YesNo = Field(description="Whether to protect from light")
    notes: str = Field(description="Additional storage notes")


class MedicationDosageGuidelines(BaseModel):
    adults: str = Field(description="Dosage guidelines for adults")
    pediatrics: str = Field(description="Dosage guidelines for pediatric patients")
    geriatrics: str = Field(description="Dosage guidelines for geriatric patients")
    max_daily_dose: str = Field(description="Maximum recommended daily dose")


class MedicationBatch(BaseModel):
    lot_number: str = Field(description="Manufacturer lot number")
    manufacture_date: str = Field(
        description='Manufacture date in YY-MM-DD format, e.g., "24-03-15"'
    )
    expiration_date: str = Field(
        description='Expiration date in YY-MM-DD format, e.g., "26-03-14"'
    )
    quantity_units: int = Field(description="Quantity in batch (units)")
    unit: str = Field(description='Unit of measure, e.g., "tablets", "ml"')


class MedicationPricing(BaseModel):
    wholesale_price: float = Field(description="Wholesale acquisition cost")
    suggested_retail_price: float = Field(description="Suggested retail price")


class Medication(BaseModel):
    medication_id: str = Field(description="Unique identifier for the medication")
    brand_name: str = Field(description="Brand name")
    generic_name: str = Field(description="Generic name")
    dosage_form: str = Field(description="Dosage form, e.g., tablet, capsule, solution")
    strength: str = Field(description='Strength, e.g., "500 mg"')
    route: str = Field(description="Route of administration, e.g., oral, IV, topical")
    ndc: str = Field(description="National Drug Code (NDC)")
    atc_code: str = Field(description="ATC classification code")
    manufacturer: str = Field(description="Manufacturer name")
    prescription_required: YesNo = Field(
        description="Whether prescription is required"
    )
    controlled_substance_schedule: ControlledSubstanceSchedule = Field(
        description='Controlled substance schedule (II, III, IV, V) or "none"'
    )
    indications: List[str] = Field(description="List of indications")
    contraindications: List[str] = Field(description="List of contraindications")
    warnings: List[str] = Field(description="List of warnings")
    common_side_effects: List[str] = Field(description="List of common side effects")
    storage: MedicationStorage = Field(description="Storage information")
    dosage_guidelines: MedicationDosageGuidelines = Field(
        description="Dosage guidelines"
    )
    images: List[MedicationImage] = Field(description="Medication images")
    batches: Dict[str, MedicationBatch] = Field(
        description="Batches indexed by batch_id"
    )
    pricing: MedicationPricing = Field(description="Pricing information")


# Patient models
class PatientAddress(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province")
    zip: str = Field(description="Postal code")


class PatientCurrentMedication(BaseModel):
    medication_id: str = Field(description="Medication ID")
    name: str = Field(description="Medication name")
    strength: str = Field(description='Strength, e.g., "500 mg"')
    dosage: str = Field(description="Dosage instructions or amount")
    start_date: str = Field(description='Start date in YY-MM-DD format')


class PatientInsurance(BaseModel):
    provider: str = Field(description="Insurance provider")
    plan_name: str = Field(description="Plan name")
    member_id: str = Field(description="Member ID")
    group: str = Field(description="Group ID")
    bin: str = Field(description="BIN number")
    pcn: str = Field(description="PCN number")


class MedicinePaymentMethodExtraInfo(BaseModel):
    brand: Optional[str] = Field(
        default=None, description="Card brand for card payments (if applicable)"
    )
    last_four: Optional[str] = Field(
        default=None, description="Last four digits (if applicable)"
    )


class MedicinePaymentMethod(BaseModel):
    source: str = Field(description="Type of payment method, e.g., credit_card")
    payment_method_id: str = Field(
        description="Unique identifier for the payment method"
    )
    extra_info: MedicinePaymentMethodExtraInfo = Field(
        description="Additional payment metadata"
    )


class EmergencyContact(BaseModel):
    name: str = Field(description="Contact's full name")
    relationship: str = Field(description="Relationship to the patient")
    phone: str = Field(description="Contact phone number")


class SavedDependent(BaseModel):
    first_name: str = Field(description="Dependent's first name")
    last_name: str = Field(description="Dependent's last name")
    dob: str = Field(description='Date of birth in YY-MM-DD format')
    relationship: str = Field(description="Relationship to the patient")


class Patient(BaseModel):
    patient_id: str = Field(description="Unique identifier for the patient")
    name: Name = Field(description="Patient's name")
    address: PatientAddress = Field(description="Patient's address")
    email: str = Field(description="Patient's email")
    phone: str = Field(description="Patient's phone number")
    dob: str = Field(description='Date of birth in YY-MM-DD format')
    gender: str = Field(description="Gender")
    allergies: List[str] = Field(description="List of allergies")
    medical_conditions: List[str] = Field(description="List of medical conditions")
    current_medications: List[PatientCurrentMedication] = Field(
        description="Current medications the patient is taking"
    )
    insurance: PatientInsurance = Field(description="Insurance information")
    payment_methods: Dict[str, MedicinePaymentMethod] = Field(
        description="Saved payment methods indexed by payment method ID"
    )
    emergency_contacts: List[EmergencyContact] = Field(
        description="List of emergency contacts"
    )
    saved_dependents: List[SavedDependent] = Field(
        description="List of saved dependents"
    )
    membership: str = Field(
        description="Pharmacy loyalty or membership program designation"
    )
    prescriptions: List[str] = Field(
        description="List of prescription IDs associated with the patient"
    )


# Prescription models
class PharmacyAddress(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    state: str = Field(description="State")
    zip: str = Field(description="Postal code")


class PharmacyInfo(BaseModel):
    pharmacy_id: str = Field(description="Unique pharmacy identifier")
    name: str = Field(description="Pharmacy name")
    address: PharmacyAddress = Field(description="Pharmacy address")


class Prescriber(BaseModel):
    doctor_id: str = Field(description="Prescribing doctor's ID")
    name: Name = Field(description="Prescriber's name")
    npi: str = Field(description="National Provider Identifier (NPI)")
    clinic: str = Field(description="Clinic or practice name")


class MedicationOrder(BaseModel):
    medication_id: str = Field(description="Medication identifier")
    brand_name: str = Field(description="Brand name")
    generic_name: str = Field(description="Generic name")
    strength: str = Field(description='Strength, e.g., "500 mg"')
    dosage_form: str = Field(description="Dosage form, e.g., tablet, capsule")
    route: str = Field(description="Route of administration")
    sig: str = Field(description="Directions for use (SIG)")
    quantity: int = Field(description="Quantity prescribed")
    days_supply: int = Field(description="Days' supply")
    substitution_allowed: YesNo = Field(
        description="Whether generic substitution is allowed"
    )
    refills_allowed: int = Field(description="Total refills allowed")
    refills_remaining: int = Field(description="Refills remaining")


class DispensedItem(BaseModel):
    medication_id: str = Field(description="Medication identifier")
    quantity_dispensed: int = Field(description="Quantity dispensed")
    lot_number: str = Field(description="Dispensed lot number")
    expiration_date: str = Field(
        description='Expiration date in YY-MM-DD format for dispensed item'
    )
    price: float = Field(description="Price charged for this item")


class DispenseInsuranceReversal(BaseModel):
    date_est: str = Field(
        description='Reversal date/time in EST in YY-MM-DD-HH-MM format'
    )
    amount: float = Field(description="Reversed amount")


class DispenseInsurance(BaseModel):
    billed_amount: float = Field(description="Amount billed to insurance")
    insurance_paid: float = Field(description="Amount paid by insurance")
    patient_copay: float = Field(description="Patient copay amount")
    prior_authorization_id: Optional[str] = Field(
        default=None, description="Prior authorization identifier, if applicable"
    )
    reversals: List[DispenseInsuranceReversal] = Field(
        description="List of claim reversals"
    )


class Dispense(BaseModel):
    fill_id: str = Field(description="Unique fill identifier")
    date_filled_est: str = Field(
        description='Date filled in EST in YY-MM-DD-HH-MM format'
    )
    pharmacist_id: str = Field(description="Pharmacist identifier")
    items: List[DispensedItem] = Field(description="Items dispensed in this fill")
    insurance: DispenseInsurance = Field(description="Insurance claim details")


class PharmacyPayment(BaseModel):
    payment_id: str = Field(description="Unique identifier for the payment")
    amount: float = Field(description="Payment amount")


class Prescription(BaseModel):
    prescription_id: str = Field(description="Unique prescription identifier")
    patient_id: str = Field(description="ID of the patient")
    pharmacy: PharmacyInfo = Field(description="Pharmacy information")
    prescriber: Prescriber = Field(description="Prescriber information")
    status: PrescriptionStatus = Field(description="Current prescription status")
    medication_orders: List[MedicationOrder] = Field(
        description="Medication orders under this prescription"
    )
    created_at: str = Field(description='Creation date in YY-MM-DD format')
    expires_at: str = Field(description='Expiration date in YY-MM-DD format')
    dispenses: List[Dispense] = Field(description="List of fills/dispenses")
    payment_history: List[PharmacyPayment] = Field(
        description="History of payments for this prescription"
    )
    total_items: int = Field(description="Total number of items prescribed")
    noncovered_items: int = Field(
        description="Number of items not covered by insurance"
    )
    counseling_offered: YesNo = Field(description="Whether counseling was offered")
    notes: str = Field(description="Additional notes")


# Medicine DB
class MedicineDB(DB):
    """Database of all medications, patients, and prescriptions."""

    medications: Dict[str, Medication] = Field(
        description="Dictionary of all medications indexed by medication ID"
    )
    patients: Dict[str, Patient] = Field(
        description="Dictionary of all patients indexed by patient ID"
    )
    prescriptions: Dict[str, Prescription] = Field(
        description="Dictionary of all prescriptions indexed by prescription ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_medications = len(self.medications)
        num_patients = len(self.patients)
        num_prescriptions = len(self.prescriptions)
        total_batches = sum(len(m.batches) for m in self.medications.values())
        total_dispenses = sum(
            len(p.dispenses) for p in self.prescriptions.values()
        )
        return {
            "num_medications": num_medications,
            "num_patients": num_patients,
            "num_prescriptions": num_prescriptions,
            "total_batches": total_batches,
            "total_dispenses": total_dispenses,
        }