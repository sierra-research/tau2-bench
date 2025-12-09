from typing import List, Optional, Dict, Any
from copy import deepcopy

from loguru import logger

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.medicine.data_model import (
    YesNo,
    ControlledSubstanceSchedule,
    PrescriptionStatus,
    Medication,
    MedicationBatch,
    MedicationImage,
    MedicationStorage,
    MedicationDosageGuidelines,
    MedicationPricing,
    Patient,
    MedicinePaymentMethod,
    MedicinePaymentMethodExtraInfo,
    PatientAddress,
    PatientInsurance,
    PharmacyInfo,
    PharmacyAddress,
    Prescriber,
    Name,
    MedicationOrder,
    DispensedItem,
    DispenseInsurance,
    DispenseInsuranceReversal,
    Dispense,
    PharmacyPayment,
    Prescription,
    MedicineDB,
)


class MedicineTools(ToolKitBase):
    """All tools for the medicine/pharmacy domain."""

    db: MedicineDB

    def __init__(self, db: MedicineDB) -> None:
        super().__init__(db)

    # --------- Internal helpers --------- #

    def _get_patient(self, patient_id: str) -> Patient:
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient {patient_id} not found")
        return self.db.patients[patient_id]

    def _get_prescription(self, prescription_id: str) -> Prescription:
        if prescription_id not in self.db.prescriptions:
            raise ValueError(f"Prescription {prescription_id} not found")
        return self.db.prescriptions[prescription_id]

    def _get_medication(self, medication_id: str) -> Medication:
        if medication_id not in self.db.medications:
            raise ValueError(f"Medication {medication_id} not found")
        return self.db.medications[medication_id]

    def _get_new_prescription_id(self) -> str:
        """Assume at most 3 created per task."""
        for pid in ["RX1001", "RX1002", "RX1003"]:
            if pid not in self.db.prescriptions:
                return pid
        raise ValueError("Too many prescriptions")

    def _get_new_fill_id(self, prescription: Prescription) -> str:
        """Generate a new fill id based on current number of fills."""
        idx = len(prescription.dispenses) + 1
        return f"FILL{idx:02d}"

    def _get_date_short(self) -> str:
        """YY-MM-DD"""
        return "24-05-15"

    def _get_datetime_short(self) -> str:
        """YY-MM-DD-HH-MM in EST"""
        return "24-05-15-15-00"

    # --------- READ tools --------- #

    @is_tool(ToolType.READ)
    def get_patient_details(self, patient_id: str) -> Patient:
        """Get full patient profile details."""
        return self._get_patient(patient_id)

    @is_tool(ToolType.READ)
    def get_prescription_details(self, prescription_id: str) -> Prescription:
        """Get full prescription details."""
        return self._get_prescription(prescription_id)

    @is_tool(ToolType.READ)
    def get_medication_details(self, medication_id: str) -> Medication:
        """Get medication master record."""
        return self._get_medication(medication_id)

    @is_tool(ToolType.READ)
    def list_patient_prescriptions(self, patient_id: str) -> List[Prescription]:
        """List all prescriptions belonging to a patient."""
        patient = self._get_patient(patient_id)
        return [self._get_prescription(pid) for pid in patient.prescriptions]

    @is_tool(ToolType.READ)
    def search_medications(
        self,
        query: str,
        prescription_required: Optional[YesNo] = None,
        controlled_substance: Optional[ControlledSubstanceSchedule] = None,
        route: Optional[str] = None,
    ) -> List[Medication]:
        """
        Search medications by brand/generic/indications and optional filters.
        """
        q = query.strip().lower()
        results: List[Medication] = []
        for med in self.db.medications.values():
            if (
                q in med.brand_name.lower()
                or q in med.generic_name.lower()
                or any(q in ind.lower() for ind in med.indications)
            ):
                if prescription_required and med.prescription_required != prescription_required:
                    continue
                if controlled_substance and med.controlled_substance_schedule != controlled_substance:
                    continue
                if route and med.route.lower() != route.lower():
                    continue
                results.append(med)
        return results

    @is_tool(ToolType.READ)
    def check_medication_inventory(self, medication_id: str) -> Dict[str, Any]:
        """
        Summarize inventory for a medication: total units and batches.
        """
        med = self._get_medication(medication_id)
        total_units = sum(b.quantity_units for b in med.batches.values())
        soonest_exp = None
        if med.batches:
            # Compare as strings YY-MM-DD lexicographically works for same century
            soonest_exp = min(b.expiration_date for b in med.batches.values())
        return {
            "medication_id": medication_id,
            "brand_name": med.brand_name,
            "generic_name": med.generic_name,
            "total_units": total_units,
            "num_batches": len(med.batches),
            "soonest_expiration": soonest_exp,
        }

    # --------- WRITE tools: Patient updates --------- #

    @is_tool(ToolType.WRITE)
    def update_patient_contact(
        self,
        patient_id: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[PatientAddress | dict] = None,
    ) -> Patient:
        """
        Update patient's contact info (email/phone/address).
        """
        patient = self._get_patient(patient_id)
        if email is not None:
            patient.email = email
        if phone is not None:
            patient.phone = phone
        if address is not None:
            if isinstance(address, dict):
                address = PatientAddress(**address)
            patient.address = address
        return patient

    @is_tool(ToolType.WRITE)
    def update_patient_insurance(
        self,
        patient_id: str,
        insurance: PatientInsurance | dict,
    ) -> Patient:
        """
        Update patient's primary insurance profile.
        """
        patient = self._get_patient(patient_id)
        if isinstance(insurance, dict):
            insurance = PatientInsurance(**insurance)
        patient.insurance = insurance
        return patient

    @is_tool(ToolType.WRITE)
    def add_patient_payment_method(
        self, patient_id: str, payment_method: MedicinePaymentMethod | dict
    ) -> Patient:
        """
        Add a saved payment method to patient profile.
        """
        patient = self._get_patient(patient_id)
        if isinstance(payment_method, dict):
            payment_method = MedicinePaymentMethod(**payment_method)
        pmid = payment_method.payment_method_id
        if pmid in patient.payment_methods:
            raise ValueError(f"Payment method {pmid} already exists")
        patient.payment_methods[pmid] = payment_method
        return patient

    @is_tool(ToolType.WRITE)
    def remove_patient_payment_method(self, patient_id: str, payment_method_id: str) -> Patient:
        """
        Remove a saved payment method from patient profile.
        """
        patient = self._get_patient(patient_id)
        if payment_method_id not in patient.payment_methods:
            raise ValueError(f"Payment method {payment_method_id} not found")
        patient.payment_methods.pop(payment_method_id)
        return patient

    # --------- WRITE tools: Prescription lifecycle --------- #

    @is_tool(ToolType.WRITE)
    def create_prescription(
        self,
        patient_id: str,
        pharmacy: PharmacyInfo | dict,
        prescriber: Prescriber | dict,
        medication_orders: List[MedicationOrder | dict],
        expires_at: str,
        notes: str = "",
        counseling_offered: YesNo = "no",
        noncovered_items: int = 0,
    ) -> Prescription:
        """
        Create a new prescription for a patient.
        """
        patient = self._get_patient(patient_id)
        if isinstance(pharmacy, dict):
            pharmacy = PharmacyInfo(**pharmacy)
        if isinstance(prescriber, dict):
            prescriber = Prescriber(**prescriber)
        if all(isinstance(mo, dict) for mo in medication_orders):
            medication_orders = [MedicationOrder(**mo) for mo in medication_orders]

        prescription_id = self._get_new_prescription_id()
        total_items = sum(mo.quantity for mo in medication_orders)

        prescription = Prescription(
            prescription_id=prescription_id,
            patient_id=patient_id,
            pharmacy=pharmacy,
            prescriber=prescriber,
            status="active",
            medication_orders=deepcopy(medication_orders),
            created_at=self._get_date_short(),
            expires_at=expires_at,
            dispenses=[],
            payment_history=[],
            total_items=total_items,
            noncovered_items=noncovered_items,
            counseling_offered=counseling_offered,
            notes=notes,
        )

        # Update DB
        self.db.prescriptions[prescription_id] = prescription
        patient.prescriptions.append(prescription_id)
        return prescription

    @is_tool(ToolType.WRITE)
    def update_prescription_status(
        self, prescription_id: str, status: PrescriptionStatus
    ) -> Prescription:
        """
        Update the status of a prescription (e.g., on-hold, canceled).
        """
        rx = self._get_prescription(prescription_id)
        rx.status = status
        return rx

    @is_tool(ToolType.WRITE)
    def transfer_prescription(
        self,
        prescription_id: str,
        new_pharmacy: PharmacyInfo | dict,
    ) -> Prescription:
        """
        Transfer prescription to a new pharmacy and mark status as 'transferred'.
        """
        rx = self._get_prescription(prescription_id)
        if isinstance(new_pharmacy, dict):
            new_pharmacy = PharmacyInfo(**new_pharmacy)
        rx.pharmacy = new_pharmacy
        rx.status = "transferred"
        return rx

    @is_tool(ToolType.WRITE)
    def set_counseling_offered(
        self, prescription_id: str, offered: YesNo
    ) -> Prescription:
        """
        Mark whether counseling was offered.
        """
        rx = self._get_prescription(prescription_id)
        rx.counseling_offered = offered
        return rx

    @is_tool(ToolType.WRITE)
    def add_prescription_payment(
        self,
        prescription_id: str,
        payment: PharmacyPayment | dict,
    ) -> Prescription:
        """
        Append a payment to prescription payment history.
        """
        rx = self._get_prescription(prescription_id)
        if isinstance(payment, dict):
            payment = PharmacyPayment(**payment)

        # Validate patient's saved payment method exists
        patient = self._get_patient(rx.patient_id)
        if payment.payment_id not in patient.payment_methods:
            raise ValueError(f"Payment method {payment.payment_id} not found for patient")

        rx.payment_history.append(payment)
        return rx

    @is_tool(ToolType.WRITE)
    def cancel_prescription(self, prescription_id: str) -> Prescription:
        """
        Cancel prescription and append negative payments as refunds.
        Note: Insurance reversal not automated here.
        """
        rx = self._get_prescription(prescription_id)

        # reverse the payments
        refunds = [
            PharmacyPayment(payment_id=pay.payment_id, amount=-pay.amount)
            for pay in rx.payment_history
            if pay.amount != 0
        ]
        rx.payment_history.extend(refunds)
        rx.status = "canceled"
        logger.warning("Insurance claim reversal not implemented in cancel_prescription.")
        return rx

    # --------- WRITE tools: Dispensing / insurance --------- #

    @is_tool(ToolType.WRITE)
    def fill_prescription(
        self,
        prescription_id: str,
        pharmacist_id: str,
        items: List[DispensedItem | dict],
        insurance: DispenseInsurance | dict,
        payments: List[PharmacyPayment | dict],
    ) -> Prescription:
        """
        Record a fill/dispense event with insurance claim and patient payment.
        Also decrements refills_remaining for dispensed medication orders,
        and decrements inventory for dispensed items by lot.
        """
        rx = self._get_prescription(prescription_id)

        if isinstance(insurance, dict):
            insurance = DispenseInsurance(**insurance)
        if all(isinstance(i, dict) for i in items):
            items = [DispensedItem(**i) for i in items]
        if all(isinstance(p, dict) for p in payments):
            payments = [PharmacyPayment(**p) for p in payments]

        # Validate items correspond to medication orders
        order_ids = {mo.medication_id for mo in rx.medication_orders}
        for it in items:
            if it.medication_id not in order_ids:
                raise ValueError(f"Dispensed item medication_id {it.medication_id} not in prescription orders")

        # Validate patient payment methods exist
        patient = self._get_patient(rx.patient_id)
        for p in payments:
            if p.payment_id not in patient.payment_methods:
                raise ValueError(f"Payment method {p.payment_id} not found for patient")

        # Compute expected patient amount
        total_item_price = sum(i.price for i in items)
        uncovered_amount = max(0.0, total_item_price - insurance.billed_amount)
        expected_patient = round(insurance.patient_copay + uncovered_amount, 2)
        actual_patient = round(sum(p.amount for p in payments), 2)
        if actual_patient != expected_patient:
            raise ValueError(
                f"Payment mismatch: expected {expected_patient} but got {actual_patient}"
            )

        # Create and append fill
        fill = Dispense(
            fill_id=self._get_new_fill_id(rx),
            date_filled_est=self._get_datetime_short(),
            pharmacist_id=pharmacist_id,
            items=deepcopy(items),
            insurance=insurance,
        )
        rx.dispenses.append(fill)

        # Append payments
        rx.payment_history.extend(deepcopy(payments))

        # Update refills_remaining for dispensed orders (if applicable)
        dispensed_ids = {i.medication_id for i in items}
        for mo in rx.medication_orders:
            if mo.medication_id in dispensed_ids and mo.refills_remaining > 0:
                mo.refills_remaining -= 1

        # Update inventory by lot number per medication
        for it in items:
            med = self._get_medication(it.medication_id)
            if it.lot_number not in med.batches:
                raise ValueError(f"Lot number {it.lot_number} not found for medication {it.medication_id}")
            batch = med.batches[it.lot_number]
            if batch.quantity_units < it.quantity_dispensed:
                raise ValueError(
                    f"Insufficient inventory in lot {it.lot_number} for medication {it.medication_id}"
                )
            # Do not enforce expiration date logic strictly here; could be added as needed.
            batch.quantity_units -= it.quantity_dispensed

        # Update status
        rx.status = "filled"
        return rx

    @is_tool(ToolType.WRITE)
    def reverse_insurance_claim(
        self,
        prescription_id: str,
        fill_id: str,
        reversal_amount: float,
        refund_to_payment_id: Optional[str] = None,
        refund_amount: Optional[float] = None,
    ) -> Prescription:
        """
        Reverse an insurance claim for a specific fill and optionally refund the patient payment.
        """
        rx = self._get_prescription(prescription_id)

        fill = next((d for d in rx.dispenses if d.fill_id == fill_id), None)
        if not fill:
            raise ValueError(f"Fill {fill_id} not found for prescription {prescription_id}")

        reversal = DispenseInsuranceReversal(
            date_est=self._get_datetime_short(),
            amount=reversal_amount,
        )
        fill.insurance.reversals.append(reversal)

        if refund_to_payment_id is not None and refund_amount is not None:
            # Validate patient payment method exists
            patient = self._get_patient(rx.patient_id)
            if refund_to_payment_id not in patient.payment_methods:
                raise ValueError(f"Payment method {refund_to_payment_id} not found for patient")
            rx.payment_history.append(
                PharmacyPayment(payment_id=refund_to_payment_id, amount=-abs(refund_amount))
            )

        # After reversal, move status back to on-hold for safety
        rx.status = "on-hold"
        return rx

    # --------- READ utility --------- #

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer to human agents if explicitly requested or cannot be solved with tools.
        """
        return "Transfer successful"


if __name__ == "__main__":
    from tau2.domains.medicine.utils import MEDICINE_DB_PATH

    med_tools = MedicineTools(MedicineDB.load(MEDICINE_DB_PATH))
    print(med_tools.get_statistics())
