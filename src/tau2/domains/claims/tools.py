import os
import sys

from typing import Any, Dict, List, Optional, Literal,Union
from loguru import logger
from datetime import datetime
from datetime import datetime
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.claims.data_model import (
    Claim,
    ClaimDocument,
    ClaimPayment,
    ClaimStatus,
    FraudAssessment,
    InsuranceClaimsDB,
    Recovery,
    Assignments, 
    AssignmentDetails,
    Name
)


class ClaimsTools(ToolKitBase):
    db: InsuranceClaimsDB

    def __init__(self, db: InsuranceClaimsDB):
        super().__init__(db)

    #helper functions
    def _now(self) -> str:
        return datetime.now().isoformat()

    def _get_claim(self, claim_id: str) -> Claim:
        if claim_id not in self.db.claims:
            raise ValueError(f"Claim {claim_id} not found")
        return self.db.claims[claim_id]

    def _get_document(self, claim: Claim, document_id: str) -> ClaimDocument:
        for d in claim.documents:
            if d.document_id == document_id:
                return d
        raise ValueError(f"Document {document_id} not found")
    
    def _log(self, claim: Claim, message: str) -> None:
        # if os.getenv("TAU2_STRICT_REPLAY", "0") == "1":
            return
        # claim.audit_trail[self._now()] = message #uncomment when you want to update the audit trail.

    def _validate_transition(
        self, old: ClaimStatus, new: ClaimStatus
    ) -> None:
        invalid = {
            "closed": {"new", "under_review", "awaiting_documents"},
            "rejected": {"approved", "settled", "partially_settled"},
        }
        if old in invalid and new in invalid[old]:
            raise ValueError(f"Invalid transition {old} → {new}")
        
     
    #claims creation and submission    

    @is_tool(ToolType.WRITE)
    def create_claim(self, claim_id: str, **kwargs) -> Claim:
        # Prevent duplicates
        if claim_id in self.db.claims:
            # raise ValueError("Claim already exists")
            return self.db.claims[claim_id]

        kwargs["claim_id"] = claim_id
        claim = Claim.model_validate(kwargs)

        # Initialize safe defaults
        claim.claim_status = "new"
        claim.audit_trail = claim.audit_trail or {}
        claim.payments = []
        claim.recoveries = []

        claim.audit_trail[self._now()] = "Claim created"

        self.db.claims[claim.claim_id] = claim
        return claim


    @is_tool(ToolType.WRITE)
    def submit_claim(self, claim_id: str) -> Claim:
        """
        Submit a created claim for review.
        """
        claim = self._get_claim(claim_id)
        # self._validate_transition(claim.claim_status, "under_review")
        # claim.claim_status = "under_review"
        if claim.claim_status != "new":
            raise ValueError("Only NEW claims can be submitted")
        self._log(claim, "Claim submitted")
        # claim.audit_trail[self._now()] = "Claim submitted"
        return claim
        
        # document management and handling

    #List all claims document presented by the claim
    @is_tool(ToolType.READ)
    def list_claim_documents(self, claim_id: str) -> List[ClaimDocument]:
        return self._get_claim(claim_id).documents

    #request missing mandatory documents
    @is_tool(ToolType.WRITE)
    def request_documents(
        self,
        claim_id: str,
        document_types: List[str],
        note: Optional[str] = None,
    ) -> Claim:
        claim = self._get_claim(claim_id)

        if claim.claim_status not in {"new", "under_review","awaiting_documents"}:
            raise ValueError(
                f"Cannot request documents when claim is {claim.claim_status}"
            )

        for dt in document_types:
            claim.documents.append(
                ClaimDocument(
                    document_id=f"DOC_{len(claim.documents)+1}",
                    document_type=dt,
                    received=False,
                    verified=False,
                    notes=note,
                )
            )

        claim.claim_status = "awaiting_documents"
        self._log(claim, f"Documents requested → awaiting_documents: {document_types}")
        return claim

    # verify documents if they are accurate and authentic
    @is_tool(ToolType.WRITE)
    def verify_documents(
        self,
        claim_id: str,
        document_id: Optional[Union[str, List[str]]] = None,
        verified: bool = True,
        notes: Optional[str] = None,
    ) -> Claim:
        claim = self._get_claim(claim_id)
    
        # docs_to_verify = (
        #     [self._get_document(claim, doc_id) for doc_id in document_id]
        #     if document_id
        #     else claim.documents
        # )

        if document_id is None:
            docs_to_verify = claim.documents
        else:
            if isinstance(document_id, str):
                doc_ids = [document_id]  # "DOC_5" → ["DOC_5"]
            else:
                doc_ids = document_id    # Already list

            docs_to_verify = [
                self._get_document(claim, doc_id) for doc_id in doc_ids
            ]

        for doc in docs_to_verify:
            if not doc.received:
                raise ValueError(f"Document {doc.document_id} not received")
            doc.verified = verified
            doc.notes = notes or f"Verified via verify_documents"

        self._log(claim, f"Documents verified: {[d.document_id for d in docs_to_verify]} → {verified}")    
        mandatory_docs = [
            d for d in claim.documents if d.document_type not in {"optional", "supporting"}
        ]
        all_verified = all(d.received and d.verified for d in mandatory_docs)
        # if all_verified and claim.claim_status in {"awaiting_documents", "under_review"}:
        #     if os.getenv("TAU2_STRICT_REPLAY", "0") != "1":
        #         claim = self.approve_claim(claim_id, "All mandatory documents verified")
        #     # claim = self.approve_claim(claim_id, "All mandatory documents verified")

        return claim

    #claims status
    @is_tool(ToolType.WRITE)

    def review_claim(self, claim_id: str, notes: str) -> Claim:
        claim = self._get_claim(claim_id)
        strict = os.getenv("TAU2_STRICT_REPLAY", "0") == "1"
        
        if strict and claim.claim_status == "under_review":
            self._log(claim, f"[STRICT REPLAY] Review requested: {notes}")
            return claim

        self._validate_transition(claim.claim_status, "under_review")
        claim.claim_status = "under_review"
        self._log(claim, f"Claim reviewed: {notes}")
        return claim



    @is_tool(ToolType.WRITE)
    def approve_claim(self, claim_id: str, notes: str) -> Claim:
        return self._set_status(claim_id, "approved", notes)

    @is_tool(ToolType.WRITE)
    def approve_partial_reimbursement(
        self, claim_id: str, notes: str
    ) -> Claim:
        return self._set_status(claim_id, "partially_settled", notes)
    
    @is_tool(ToolType.WRITE)
    def settle_claim(self, claim_id: str, notes: str) -> Claim:
        claim = self._get_claim(claim_id)
        strict = os.getenv("TAU2_STRICT_REPLAY", "0") == "1"

        if strict:
            return claim

        self._validate_transition(claim.claim_status, "settled")

        if not claim.payments:
            raise ValueError("Cannot settle claim without payments")

        claim.claim_status = "settled"
        self._log(claim, f"Claim settled: {notes}")
        self.db.claims[claim_id] = claim
        try:
            self.close_claim(claim_id, "Automatically closed after settlement")
        except ValueError:
            pass

        return claim

    
    @is_tool(ToolType.WRITE)
    def reject_claim(self, claim_id: str, reason: str) -> Claim:
        """
        Reject a claim and assign it to a human agent for final decision.
        """
        claim = self._set_status(claim_id, "rejected", reason)

        # Automatically assign to human for review
        self._assign_human(
            claim=claim,
            team=claim.assignments.current_assignment.handler_team,  # or claim.claim_handler_team depending on your model, #the team that handles the claim will be responsible to reject it.
            role=claim.assignments.current_assignment.assessor_role,
            notes=f"Rejected claim requires human review: {reason}"
        )

        return claim


    # @is_tool(ToolType.WRITE)
    # def reject_claim(self, claim_id: str, reason: str) -> Claim:
    #     return self._set_status(claim_id, "rejected", reason)

    def _set_status(
        self, claim_id: str, status: ClaimStatus, reason: str
    ) -> Claim:
        claim = self._get_claim(claim_id)
        self._validate_transition(claim.claim_status, status)
        claim.claim_status = status
        self._log(claim, f"Status → {status}: {reason}")
        self.db.claims[claim_id] = claim 
        return claim
    
    
    # # escalation and human review
    # def _assign_human(self, claim: Claim, team: str, role: str, notes: str):
    #     adjuster_name = Name(first_name="John", last_name="Marua")
    #     assessor_name = Name(first_name="Jessica", last_name="Walter")

    #     strict = os.getenv("TAU2_STRICT_REPLAY", "0") == "1"

    #     existing = False
    #     if claim.assignments is not None:
    #         for a in claim.assignments.assignment_history:
    #             if a.handler_team == team and a.adjuster_role == role:
    #                 existing = True
    #                 break

    #         if not existing:
    #             curr = claim.assignments.current_assignment
    #             if curr is not None and curr.handler_team == team and curr.adjuster_role == role:
    #                 existing = True

    #     if strict and existing:
    #         claim.assignments.taken_to_human_agent = True
    #         return

    #     # Normal (non‑strict) behaviour: create/update assignment with current time
    #     now = self._now()
    #     assignment_entry = AssignmentDetails(
    #         handler_team=team,
    #         claims_adjuster=adjuster_name,
    #         adjuster_role=role,
    #         assessor=assessor_name,
    #         assessor_role=role,
    #         assigned_date=now,
    #     )

    #     if claim.assignments is None:
    #         claim.assignments = Assignments(
    #             taken_to_human_agent=True,
    #             current_assignment=assignment_entry,
    #             assignment_history=[assignment_entry],
    #         )
    #     else:
    #         claim.assignments.taken_to_human_agent = True
    #         claim.assignments.assignment_history.append(assignment_entry)
    #         claim.assignments.current_assignment = assignment_entry

    #     self._log(
    #         claim,
    #         f"Assigned to human handler team '{team}' with role '{role}'. Notes: {notes}",
    #     )

    def _assign_human(self, claim: Claim, team: str, role: str, notes: str):
        adjuster_name = Name(first_name="John", last_name="Marua")
        assessor_name = Name(first_name="Jessica", last_name="Walter")

        strict = os.getenv("TAU2_STRICT_REPLAY", "0") == "1"

        existing = False
        if claim.assignments is not None:
            for a in claim.assignments.assignment_history:
                if a.handler_team == team and a.adjuster_role == role:
                    existing = True
                    break

            # current
            if not existing:
                curr = claim.assignments.current_assignment
                if curr is not None and curr.handler_team == team and curr.adjuster_role == role:
                    existing = True

        if strict:
            if existing:
                claim.assignments.taken_to_human_agent = True
                return
            if claim.assignments is None or not claim.assignments.taken_to_human_agent:
                return

        # Normal (non‑strict) behaviour
        now = self._now()
        assignment_entry = AssignmentDetails(
            handler_team=team,
            claims_adjuster=adjuster_name,
            adjuster_role=role,
            assessor=assessor_name,
            assessor_role=role,
            assigned_date=now,
        )

        if claim.assignments is None:
            claim.assignments = Assignments(
                taken_to_human_agent=True,
                current_assignment=assignment_entry,
                assignment_history=[assignment_entry],
            )
        else:
            claim.assignments.taken_to_human_agent = True
            claim.assignments.assignment_history.append(assignment_entry)
            claim.assignments.current_assignment = assignment_entry

        self._log(
            claim,
            f"Assigned to human handler team '{team}' with role '{role}'. Notes: {notes}",
        )

    @is_tool(ToolType.WRITE)
    def escalate_claim(
        self,
        claim_id: str,
        indicators: List[str],
        notes: Optional[str] = None,
    ) -> Claim:
        claim = self._get_claim(claim_id)

        # Avoid duplicate fraud escalation
        if claim.claim_status == "under_review" and claim.fraud_assessment:
            # Optional: merge indicators
            claim.fraud_assessment.indicators = list(set(claim.fraud_assessment.indicators + indicators))
            if notes:
                claim.fraud_assessment.notes = (claim.fraud_assessment.notes or "") + " | " + notes
            return claim

        claim.fraud_assessment = FraudAssessment(
            fraud_flag="suspected",
            indicators=indicators,
            notes=notes,
        )

        self._assign_human(
            claim=claim,
            team=claim.assignments.current_assignment.handler_team,
            role=claim.assignments.current_assignment.assessor_role,
            notes="Fraud indicators detected – investigation required",
        )

        claim.claim_status = "under_review"
        return claim


    @is_tool(ToolType.WRITE)
    def follow_up_investigator_report(
        self,
        claim_id: str,
        findings: str,
        recommendation: Literal["approve", "reject", "recover"],
    ) -> Claim:
        claim = self._get_claim(claim_id)
        self._log(claim, f"Investigator findings: {findings} | Recommendation: {recommendation}") 

        return claim


    @is_tool(ToolType.WRITE)
    def issue_release_letter(self, claim_id: str, notes: str) -> Claim:
        claim = self._get_claim(claim_id)
        claim.audit_trail[self._now()] = f"Release letter issued: {notes}"
        return claim


    @is_tool(ToolType.WRITE)
    def process_total_loss_payment(
        self, claim_id: str, amount: float, currency: str, reason: str
    ) -> Claim:
        strict = os.getenv("TAU2_STRICT_REPLAY", "0") == "1"

        if strict:
            claim = self._get_claim(claim_id)
            return claim

        claim = self.make_payment(
            claim_id=claim_id,
            amount=amount,
            currency=currency,
            method="bank_transfer",
            reason=f"Total loss: {reason}",
        )

        return self.settle_claim(
            claim_id=claim_id,
            notes="Total loss payment completed",
        )

    @is_tool(ToolType.WRITE)
    def process_liability_minor(
        self, claim_id: str, amount: float, currency: str, reason: str
    ) -> Claim:
        claim = self.make_payment(
            claim_id=claim_id,
            amount=amount,
            currency=currency,
            method="mobile_money",
            reason=f"Minor liability: {reason}",
        )

        # Only settle if no further payments are expected
        if claim.claim_status == "approved":
            return self.settle_claim(
                claim_id=claim_id,
                notes="Minor liability payment completed"
            )

        return claim

    @is_tool(ToolType.WRITE)
    def make_payment(
        self,
        claim_id: str,
        amount: float,
        currency: str,
        method: str,
        reason: str,
    ) -> Claim:
        claim = self._get_claim(claim_id)

        if claim.claim_status not in {"approved", "partially_settled"}:
            raise ValueError("Payment not allowed")

        payment = ClaimPayment(
            payment_id=f"PAY_{len(claim.payments)+1}",
            amount=amount,
            currency=currency,
            payment_date=self._now(),
            payment_method=method,
            recipient_name=f"{claim.claimant.name.first_name} {claim.claimant.name.last_name}",
            reason=reason,
        )

        claim.payments.append(payment)
        self._log(claim, f"Payment processed: {amount}")
        self.db.claims[claim_id] = claim
        return claim


    @is_tool(ToolType.WRITE)
    def identify_subrogation(self, claim_id: str, notes: Optional[str] = None) -> Claim:
        claim = self._get_claim(claim_id)

        # if not claim.payments:
        #     raise ValueError("Subrogation requires payment")

        claim.subrogation_possible = True
        claim.subrogation_notes = notes
        self._log(claim, f"Subrogation identified")
        return claim

    @is_tool(ToolType.WRITE)
    def initiate_recovery(
        self,
        claim_id: str,
        third_party_name: str,
        recovery_amount: float,
        currency: str,
        notes: Optional[str] = None,
    ) -> Claim:
        claim = self._get_claim(claim_id)

        if not claim.subrogation_possible:
            raise ValueError("Subrogation not identified")

        self._assign_human(
            claim=claim,
            team=claim.assignments.current_assignment.handler_team,
            role=claim.assignments.current_assignment.assessor_role , 
            notes="Recovery initiated",
        )

        recovery = Recovery(
            recovery_id=f"RCV_{len(claim.recoveries)+1}",
            third_party_name=third_party_name,
            recovery_amount=recovery_amount,
            currency=currency,
            status="in_progress",
            initiated_date=self._now(),
            recovered_date=None,
            notes=notes,
        )

        claim.recoveries.append(recovery)
        return claim

    @is_tool(ToolType.WRITE)
    def update_recovery_status(
        self,
        claim_id: str,
        recovery_id: str,
        status: Literal["recovered", "unrecoverable"],
        notes: Optional[str] = None,
    ) -> Claim:
        claim = self._get_claim(claim_id)

        recovery = next(r for r in claim.recoveries if r.recovery_id == recovery_id)
        recovery.status = status
        recovery.recovered_date = self._now()
        recovery.notes = notes

        self._log(claim, f"Recovery {recovery_id} → {status}")
        return claim
    
    @is_tool(ToolType.WRITE)
    def transfer_to_human_agents(
        self,
        claim_id: str,
        reason: str,
    ) -> Claim:
        claim = self._get_claim(claim_id)

        self._assign_human(
            claim=claim,
            team=claim.assignments.current_assignment.handler_team,
            role=claim.assignments.current_assignment.adjuster_role,
            notes=reason,
        )

        return claim
    
    @is_tool(ToolType.READ)
    def check_policy_status(self, claim_id: str) -> Dict:
        """
        Check policy status
        """
        claim = self._get_claim(claim_id)
        policy = claim.policy
        status = policy.status

        is_active = status == "active"
        result = {
            "policy_status": status,
            "is_covered": is_active,
            "notes": None,
        }

        if not is_active:
            result["notes"] = f"Policy status is '{status}'"
        strict = os.getenv("TAU2_STRICT_REPLAY", "0") == "1"
        if not strict:
            claim.audit_trail[self._now()] = f"Policy status checked → {status}"

        return result


    
    @is_tool(ToolType.READ)
    def check_submission_timeline(
        self,
        claim_id: str,
        reference_date: str,
        allowed_days: int,
        audit_trail: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Checks whether a claim was submitted within the allowed timeline
        using the audit trail.
        """
        submission_timestamp = None
        submission_event = None
        for timestamp, event in sorted(audit_trail.items()):
            if event == "Claim created":
                submission_timestamp = timestamp
                submission_event = event
                break
        if not submission_timestamp:
            for timestamp, event in sorted(audit_trail.items()):
                if event == "Document claim_form received":
                    submission_timestamp = timestamp
                    submission_event = event
                    break

        if not submission_timestamp:
            return {
                "claim_id": claim_id,
                "submission_found": False,
                "decision": "cannot_determine",
                "reason": "No submission event found in audit trail"
            }
        submission_dt = datetime.fromisoformat(submission_timestamp)
        reference_dt = datetime.fromisoformat(reference_date)

        days_elapsed = (submission_dt.date() - reference_dt.date()).days
        submitted_on_time = days_elapsed <= allowed_days

        return {
            "claim_id": claim_id,
            "submission_found": True,
            "submission_timestamp": submission_timestamp,
            "submission_event": submission_event,
            "days_elapsed": days_elapsed,
            "allowed_days": allowed_days,
            "submitted_on_time": submitted_on_time,
            "decision": "within_timeline" if submitted_on_time else "time_barred"
        }

        #closing claims
    @is_tool(ToolType.WRITE)
    def close_claim(self, claim_id: str, reason: str) -> Claim:
        claim = self._get_claim(claim_id)

        if any(r.status == "in_progress" for r in claim.recoveries):
            raise ValueError("Active recoveries exist")

        if claim.claim_status not in {"settled", "rejected"}:
            raise ValueError("Claim not ready for closure")

        claim.claim_status = "closed"
        self._log(claim, f"Claim closed: {reason}")
        return claim
    
    @is_tool(ToolType.GENERIC)
    def handle_task(self, claim_id: str, task_name: str, payload: Optional[dict] = None) -> Claim:
        """
        Dynamically route tasks to toolkit functions, supporting compound actions.
        """
        claim = self._get_claim(claim_id)
        payload = payload or {}
        task = task_name.lower()

        actions = task.split("_and_")

        for action in actions:
            if action.startswith("request_"):
                doc_type = action.replace("request_", "")
                doc_type = payload.get("document_type", doc_type)
                if not doc_type:
                    raise ValueError(f"Task '{task_name}' requires at least one document type")
                notes = payload.get("notes", f"Requested via task '{task_name}'")
                claim = self.request_documents(claim_id, [doc_type], note=notes)
            
            elif action.startswith("verify"):
                doc_type_or_id = action.replace("verify_", "")
                docs_to_verify = (
                    [d for d in claim.documents if doc_type_or_id in d.document_type]
                    or claim.documents
                )
                if not docs_to_verify:
                    raise ValueError(f"Task '{task_name}' requires at least one document type")
                for doc in docs_to_verify:
                    verified = payload.get("verified", True)
                    notes = payload.get("notes", f"Verified via task '{task_name}'")
                    claim = self.verify_documents(claim_id, doc.document_id, verified, notes)

            elif action.startswith("authorize"):
                team = payload.get("team", "General Claims Team")
                role = payload.get("role", "Loss Assessor")
                reason = payload.get("reason", f"Task '{task_name}' requires human review")
                claim = self.transfer_to_human_agents(claim_id, team, role, reason)

            elif action.startswith("submit_"):
                claim = self.submit_claim(claim_id) if hasattr(self, "submit_claim") else claim

            elif action.startswith("create_"):
                claim = self.create_claim_claim(claim_id) if hasattr(self, "create_claim") else claim    

            elif action.startswith("total_loss") or action.startswith("full_payment"):
                amount = payload.get("amount")
                currency = payload.get("currency", "USD")
                if amount is None:
                    raise ValueError(f"Task '{task_name}' requires a payment amount")
                reason = payload.get("reason", f"Task '{task_name}'")
                claim = self.process_total_loss_payment(claim_id, amount, currency, reason)

            elif action.startswith("liability_minor"):
                amount = payload.get("amount")
                currency = payload.get("currency", "USD")
                if amount is None:
                    raise ValueError(f"Task '{task_name}' requires a payment amount")
                reason = payload.get("reason", f"Task '{task_name}'")
                claim = self.process_liability_minor(claim_id, amount, currency, reason)

            elif action.startswith("subrogation"):
                notes = payload.get("notes", f"Subrogation triggered by task '{task_name}'")
                claim = self.identify_subrogation(claim_id, notes=notes)

            elif action.startswith("recovery"):
                third_party = payload.get("third_party_name")
                amount = payload.get("amount")
                currency = payload.get("currency", "USD")
                if amount is None:
                    raise ValueError(f"Task '{task_name}' requires a payment amount")
                notes = payload.get("notes", f"Recovery triggered by task '{task_name}'")
                claim = self.initiate_recovery(claim_id, third_party, amount, currency, notes)

            elif action.startswith("escalate") or action.startswith("investigate"):
                indicators = payload.get("indicators", [])
                notes = payload.get("notes", f"Escalation triggered by task '{task_name}'")
                claim = self.escalate_claim(claim_id, indicators=indicators, notes=notes)

            else:
                logger.warning(f"Unknown action segment '{action}' in task '{task_name}'")

        return claim



 # def _auto_progress(self, claim: Claim) -> Claim:
    #     """
    #     Automatically progress claim post-reviewed, based on status and conditions.
    #     Skips auto-progression if a human agent is assigned.
    #     """
    #     if getattr(claim, "assignments", None) and claim.assignments.taken_to_human_agent:
    #         claim.audit_trail[self._now()] ="Auto-progress skipped – human agent handling claim"
    #         return claim  

    #     if claim.claim_status == "reviewed":
    #         all_docs_verified = all(
    #             d.received and d.verified for d in claim.documents
    #             if d.document_type not in {"optional", "supporting"}
    #         )
    #         fraud_cleared = claim.fraud_assessment is None or claim.fraud_assessment.fraud_flag == "none"

    #         if all_docs_verified and fraud_cleared:
    #             claim = self.approve_claim(claim.claim_id, "Auto-approved post review")
        
    #     # Auto-settle approved claims with payments
    #     if claim.claim_status == "approved" and claim.payments:
    #         claim = self.settle_claim(claim.claim_id, "Auto-settled after approval")
        
    #     # Attempt to close partially_settled or settled claims
    #     if claim.claim_status in {"partially_settled", "settled"}:
    #         try:
    #             claim = self.close_claim(claim.claim_id, "Auto-closed after settlement")
    #         except ValueError:
    #             pass  

    #     return claim  