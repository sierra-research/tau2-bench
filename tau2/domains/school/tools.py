from copy import deepcopy
from typing import List, Optional, Dict

from loguru import logger

from tau2.domains.school.data_model import (
    Advisor,
    Course,
    CourseDateStatus,
    FinancialAid,
    GradingOption,
    MeetingPattern,
    Payment,
    Registration,
    RegistrationCourse,
    SchoolDB,
    Student,
    StudyLevel,
    YesNo,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class SchoolTools(ToolKitBase):
    """Frequently-used tools for the school domain."""

    db: SchoolDB

    # ----------------------
    # Helpers / Internals
    # ----------------------
    def __init__(self, db: SchoolDB) -> None:
        super().__init__(db)

    def _get_student(self, student_id: str) -> Student:
        if student_id not in self.db.students:
            raise ValueError(f"Student {student_id} not found")
        return self.db.students[student_id]

    def _get_course(self, course_id: str) -> Course:
        if course_id not in self.db.courses:
            raise ValueError(f"Course {course_id} not found")
        return self.db.courses[course_id]

    def _get_registration(self, registration_id: str) -> Registration:
        if registration_id not in self.db.registrations:
            raise ValueError(f"Registration {registration_id} not found")
        return self.db.registrations[registration_id]

    def _get_new_registration_id(self) -> str:
        """Generate a new registration ID. Assume at most 3 new registrations per task."""
        for rid in ["REG001", "REG002", "REG003"]:
            if rid not in self.db.registrations:
                return rid
        raise ValueError("Too many registrations created in this session")

    def _get_datetime_short(self) -> str:
        """Return a created_at date string matching the model format YY-MM-DD."""
        # Keep static for deterministic behavior in tests
        return "25-05-15"

    def _per_credit_rate(self, study_level: StudyLevel) -> int:
        # Example per-credit rates; adjust as needed.
        rates = {
            "undergraduate": 500,
            "graduate": 800,
            "continuing_ed": 400,
        }
        return rates[study_level]

    def _compute_course_tuition(self, course: Course, study_level: StudyLevel) -> int:
        return course.credits * self._per_credit_rate(study_level)

    def _build_registration_course(
        self, course: Course, grading_option: GradingOption, study_level: StudyLevel
    ) -> RegistrationCourse:
        return RegistrationCourse(
            course_id=course.course_id,
            course_code=course.course_code,
            section="001",
            title=course.title,
            credits=course.credits,
            grading_option=grading_option,
            tuition=self._compute_course_tuition(course, study_level),
        )

    def _credits_to_status(self, total_credits: int) -> str:
        return "full-time" if total_credits >= 12 else "part-time"

    def _overload_credits(self, total_credits: int) -> int:
        standard_load = 18
        return max(0, total_credits - standard_load)

    def _parse_time_hhmm(self, hhmm: str) -> int:
        # "HH-MM" -> minutes since midnight
        parts = hhmm.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid time format: {hhmm}")
        h = int(parts[0])
        m = int(parts[1])
        return h * 60 + m

    def _times_overlap(self, s1: str, e1: str, s2: str, e2: str) -> bool:
        a1 = self._parse_time_hhmm(s1)
        b1 = self._parse_time_hhmm(e1)
        a2 = self._parse_time_hhmm(s2)
        b2 = self._parse_time_hhmm(e2)
        return a1 < b2 and a2 < b1

    def _meeting_conflict(self, m1: MeetingPattern, m2: MeetingPattern) -> bool:
        # Conflict if any day overlaps and times overlap
        set_days = set(m1.days).intersection(set(m2.days))
        if not set_days:
            return False
        return self._times_overlap(m1.start_time_est, m1.end_time_est, m2.start_time_est, m2.end_time_est)

    def _check_time_conflicts(self, courses: List[Course]) -> None:
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                if self._meeting_conflict(courses[i].meeting_pattern, courses[j].meeting_pattern):
                    raise ValueError(
                        f"Schedule conflict between courses {courses[i].course_code} and {courses[j].course_code}"
                    )

    def _validate_payments_exist(self, student: Student, payments: List[Payment]) -> None:
        for p in payments:
            if p.payment_id not in student.payment_methods:
                raise ValueError(f"Payment method {p.payment_id} not found for student {student.student_id}")

    def _health_insurance_fee(self, health_insurance: YesNo) -> int:
        return 300 if health_insurance == "yes" else 0

    # ----------------------
    # Read tools
    # ----------------------
    @is_tool(ToolType.READ)
    def get_student_details(self, student_id: str) -> Student:
        """
        Get the details of a student.
        """
        return self._get_student(student_id)

    @is_tool(ToolType.READ)
    def get_course_details(self, course_id: str) -> Course:
        """
        Get the details of a course.
        """
        return self._get_course(course_id)

    @is_tool(ToolType.READ)
    def get_registration_details(self, registration_id: str) -> Registration:
        """
        Get the details of a registration.
        """
        return self._get_registration(registration_id)

    @is_tool(ToolType.READ)
    def list_departments(self) -> List[str]:
        """
        List all unique departments.
        """
        return sorted({c.department for c in self.db.courses.values()})

    @is_tool(ToolType.READ)
    def list_courses_by_term(self, term: str) -> List[Course]:
        """
        List all courses for a given term.
        """
        return [{
            'course_id': c.course_id,
            'course_code': c.course_code,
            'title': c.title
        } for c in self.db.courses.values() if c.term == term]

    @is_tool(ToolType.READ)
    def search_courses(
        self,
        term: str,
        department: Optional[str] = None,
        campus: Optional[str] = None,
        day: Optional[str] = None,  # DayOfWeek
        start_after: Optional[str] = None,  # "HH-MM"
        end_before: Optional[str] = None,  # "HH-MM"
        open_only: bool = False,
    ) -> List[Course]:
        """
        Search for courses with common filters.

        Args:
            term: Term in the format YYYY-TERM.
            department: Optional department code to filter.
            campus: Optional campus name to filter.
            day: Optional day of week filter (e.g., "Monday").
            start_after: Optional earliest start time ("HH-MM").
            end_before: Optional latest end time ("HH-MM").
            open_only: If True, only return courses with open seats.
        """
        results = []
        for c in self.db.courses.values():
            if c.term != term:
                continue
            if department and c.department != department:
                continue
            if campus and c.location.campus != campus:
                continue
            if day and day not in c.meeting_pattern.days:
                continue
            if start_after and self._parse_time_hhmm(c.meeting_pattern.start_time_est) < self._parse_time_hhmm(start_after):
                continue
            if end_before and self._parse_time_hhmm(c.meeting_pattern.end_time_est) > self._parse_time_hhmm(end_before):
                continue
            if open_only and not (c.current_enrollment < c.capacity):
                continue
            results.append(c)
        return results

    @is_tool(ToolType.READ)
    def get_course_meeting_status(self, course_id: str, date: str) -> CourseDateStatus:
        """
        Get the per-date meeting status for a course.

        Args:
            course_id: Course ID.
            date: Date key in the format YY-MM-DD.
        """
        course = self._get_course(course_id)
        if date not in course.dates:
            raise ValueError(f"No record for course {course_id} on date {date}")
        return course.dates[date]

    # ----------------------
    # Write tools
    # ----------------------
    @is_tool(ToolType.WRITE)
    def register_courses(
        self,
        student_id: str,
        term: str,
        course_ids: List[str],
        study_level: StudyLevel,
        grading_options: Optional[Dict[str, GradingOption]] = None,
        health_insurance: YesNo = "no",
        payment_methods: List[Payment | dict] = [],
    ) -> Registration:
        """
        Create a new registration for a student for a given term.

        - Checks capacity and time conflicts.
        - Computes tuition and validates payments equal total charges.
        - Increments course enrollments on success.

        Args:
            student_id: The ID of the student.
            term: Term in the format YYYY-TERM.
            course_ids: List of course IDs to register.
            study_level: Study level ("undergraduate", "graduate", "continuing_ed").
            grading_options: Optional map from course_id to grading option ("letter", "pass/fail", "audit"). Defaults to "letter".
            health_insurance: "yes" or "no".
            payment_methods: List of Payment objects or dicts with payment_id and amount.

        Returns:
            The created Registration.
        """
        if all(isinstance(p, dict) for p in payment_methods):
            payment_methods = [Payment(**p) for p in payment_methods]  # type: ignore

        student = self._get_student(student_id)
        courses = [self._get_course(cid) for cid in course_ids]

        # Term validation
        for c in courses:
            if c.term != term:
                raise ValueError(f"Course {c.course_id} term {c.term} does not match requested term {term}")

        # Capacity check
        for c in courses:
            if c.current_enrollment >= c.capacity:
                raise ValueError(f"Course {c.course_code} is full")

        # Time conflict check
        self._check_time_conflicts(courses)

        # Build registration courses
        reg_courses: List[RegistrationCourse] = []
        total_credits = 0
        tuition_total = 0
        for c in courses:
            go = "letter"
            if grading_options and c.course_id in grading_options:
                go = grading_options[c.course_id]
            rc = self._build_registration_course(c, go, study_level)
            reg_courses.append(rc)
            total_credits += rc.credits
            tuition_total += rc.tuition

        # Add fees
        insurance_fee = self._health_insurance_fee(health_insurance)
        total_due = tuition_total + insurance_fee

        # Validate payments
        if not isinstance(payment_methods, list):
            raise ValueError("payment_methods must be a list")
        # type: ignore
        self._validate_payments_exist(student, payment_methods)  # type: ignore
        total_paid = sum(p.amount for p in payment_methods)  # type: ignore
        if total_paid != total_due:
            raise ValueError(f"Payment amount does not add up, total due is {total_due}, but paid {total_paid}")

        # Create registration
        registration_id = self._get_new_registration_id()
        campus = courses[0].location.campus if courses else "Main"
        status = self._credits_to_status(total_credits)
        overload = self._overload_credits(total_credits)
        registration = Registration(
            registration_id=registration_id,
            student_id=student_id,
            campus=campus,
            program=f"{student.program.degree} {student.program.major}",
            study_level=study_level,
            term=term,
            status=status,  # full-time / part-time
            courses=deepcopy(reg_courses),
            advisors=[],  # None assigned by default
            payment_history=deepcopy(payment_methods),  # type: ignore
            created_at=self._get_datetime_short(),
            total_credits=total_credits,
            overload_credits=overload,
            financial_aid=FinancialAid(scholarships=0, grants=0, loans=0),
            health_insurance=health_insurance,
        )

        # Update DB: increment enrollments, save registration, link to student
        for c in courses:
            c.current_enrollment += 1

        self.db.registrations[registration_id] = registration
        self.db.students[student_id].registrations.append(registration_id)
        return registration

    @is_tool(ToolType.WRITE)
    def update_registration_courses(
        self,
        registration_id: str,
        add_course_ids: List[str],
        drop_course_ids: List[str],
        grading_options: Optional[Dict[str, GradingOption]] = None,
        payment_id: str = "",
    ) -> Registration:
        """
        Add and/or drop courses from an existing registration, handling tuition differences.

        - Checks capacity for added courses and schedule conflicts against the final schedule.
        - Adjusts course enrollments in the database.
        - Computes tuition difference and records a payment (positive charge) or refund (negative charge).

        Args:
            registration_id: Existing registration ID.
            add_course_ids: Course IDs to add.
            drop_course_ids: Course IDs to drop.
            grading_options: Optional map from course_id to grading option.
            payment_id: Student's saved payment method ID to use for additional charges.

        Returns:
            The updated Registration.
        """
        reg = self._get_registration(registration_id)
        student = self._get_student(reg.student_id)

        # Current and target course sets
        current_ids = [rc.course_id for rc in reg.courses]
        target_ids = [cid for cid in current_ids if cid not in set(drop_course_ids)]
        for cid in add_course_ids:
            if cid not in target_ids:
                target_ids.append(cid)

        # Load courses
        current_courses = [self._get_course(cid) for cid in current_ids]
        target_courses = [self._get_course(cid) for cid in target_ids]

        # Validate term consistency
        for c in target_courses:
            if c.term != reg.term:
                raise ValueError(f"Course {c.course_id} term {c.term} does not match registration term {reg.term}")

        # Capacity check for added courses
        for cid in add_course_ids:
            c = self._get_course(cid)
            if c.current_enrollment >= c.capacity:
                raise ValueError(f"Course {c.course_code} is full")

        # Time conflict check against the final schedule
        self._check_time_conflicts(target_courses)

        # Compute old and new tuition totals
        old_tuition = sum(rc.tuition for rc in reg.courses)
        old_credits = sum(rc.credits for rc in reg.courses)

        new_reg_courses: List[RegistrationCourse] = []
        new_tuition = 0
        new_credits = 0

        for c in target_courses:
            go = "letter"
            if grading_options and c.course_id in grading_options:
                go = grading_options[c.course_id]
            # study_level is stored in registration
            rc = self._build_registration_course(c, go, reg.study_level)
            new_reg_courses.append(rc)
            new_tuition += rc.tuition
            new_credits += rc.credits

        # Tuition difference (insurance fee unchanged)
        tuition_diff = new_tuition - old_tuition

        # Create payment/refund entry if needed
        if tuition_diff != 0:
            if tuition_diff > 0:
                if payment_id == "":
                    raise ValueError("payment_id is required for additional charges")
                if payment_id not in student.payment_methods:
                    raise ValueError(f"Payment method {payment_id} not found for student {student.student_id}")
                reg.payment_history.append(Payment(payment_id=payment_id, amount=tuition_diff))
            else:
                # Refund
                refund_payment_id = payment_id or (reg.payment_history[-1].payment_id if reg.payment_history else "adjustment")
                reg.payment_history.append(Payment(payment_id=refund_payment_id, amount=tuition_diff))

        # Update enrollments: decrement dropped, increment added
        dropped = set(drop_course_ids)
        added = set(add_course_ids)
        for c in current_courses:
            if c.course_id in dropped:
                c.current_enrollment = max(0, c.current_enrollment - 1)
        for c in target_courses:
            if c.course_id in added:
                c.current_enrollment += 1

        # Update registration record
        reg.courses = new_reg_courses
        reg.total_credits = new_credits
        reg.overload_credits = self._overload_credits(new_credits)
        reg.status = self._credits_to_status(new_credits)

        return reg

    @is_tool(ToolType.WRITE)
    def cancel_registration(self, registration_id: str) -> Registration:
        """
        Cancel a registration:
        - Issues refunds for all recorded payments (as negative amounts).
        - Drops all courses and updates enrollments.
        - Sets total credits and overload credits to 0.

        Note: Registration remains in the system with empty course list.
        """
        reg = self._get_registration(registration_id)
        logger.debug(reg.model_dump_json(indent=2))

        # Refund all payments
        refunds = [Payment(payment_id=p.payment_id, amount=-p.amount) for p in reg.payment_history]
        reg.payment_history.extend(refunds)

        # Adjust enrollments
        for rc in reg.courses:
            course = self._get_course(rc.course_id)
            course.current_enrollment = max(0, course.current_enrollment - 1)

        # Clear courses, reset credits
        reg.courses = []
        reg.total_credits = 0
        reg.overload_credits = 0
        reg.status = self._credits_to_status(0)

        logger.debug(self._get_registration(registration_id).model_dump_json(indent=2))
        return reg

    @is_tool(ToolType.WRITE)
    def pay_registration(
        self,
        registration_id: str,
        payment: Payment | dict,
    ) -> Registration:
        """
        Record a payment (or refund with negative amount) for a registration.

        Args:
            registration_id: The registration ID.
            payment: Payment object or dict with payment_id and amount.

        Returns:
            The updated registration.
        """
        if isinstance(payment, dict):
            payment = Payment(**payment)

        reg = self._get_registration(registration_id)
        student = self._get_student(reg.student_id)

        if payment.payment_id not in student.payment_methods:
            raise ValueError(f"Payment method {payment.payment_id} not found for student {student.student_id}")

        reg.payment_history.append(payment)
        return reg


if __name__ == "__main__":
    from tau2.domains.school.data_model import get_db

    school = SchoolTools(get_db())
    print(school.get_statistics())
