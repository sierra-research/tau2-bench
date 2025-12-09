from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.environment.db import DB


# Common enums
DayOfWeek = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
YesNo = Literal["yes", "no"]
StudyLevel = Literal["undergraduate", "graduate", "continuing_ed"]
EnrollmentStatus = Literal["full-time", "part-time"]
GradingOption = Literal["letter", "pass/fail", "audit"]


# Shared/basic types
class Name(BaseModel):
    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


class StudentAddress(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


class Payment(BaseModel):
    payment_id: str = Field(description="Unique identifier for the payment")
    amount: int = Field(description="Payment amount in dollars")


# Course-related models
class Instructor(BaseModel):
    first_name: str = Field(description="Instructor's first name")
    last_name: str = Field(description="Instructor's last name")
    email: str = Field(description="Instructor's email address")


class Location(BaseModel):
    campus: str = Field(description="Campus name")
    building: str = Field(description="Building name")
    room: str = Field(description="Room identifier")


class MeetingPattern(BaseModel):
    days: List[DayOfWeek] = Field(description="Days of the week the class meets")
    start_time_est: str = Field(description="Class start time in EST in the format HH-MM, e.g. 09-30")
    end_time_est: str = Field(description="Class end time in EST in the format HH-MM, e.g. 10-45")


class CourseDateStatusHeld(BaseModel):
    status: Literal["held"] = Field(description="Indicates the class was held")
    topic: str = Field(description="Class topic for the day")
    actual_start_time_est: str = Field(description="Actual start time in EST in the format YY-MM-DD-HH-MM")
    actual_end_time_est: str = Field(description="Actual end time in EST in the format YY-MM-DD-HH-MM")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class CourseDateStatusCanceled(BaseModel):
    status: Literal["canceled"] = Field(description="Indicates the class was canceled")
    topic: str = Field(description="Class topic for the day")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class CourseDateStatusRescheduled(BaseModel):
    status: Literal["rescheduled"] = Field(description="Indicates the class was rescheduled")
    topic: str = Field(description="Class topic for the day")
    actual_start_time_est: str = Field(description="Rescheduled start time in EST in the format YY-MM-DD-HH-MM")
    actual_end_time_est: str = Field(description="Rescheduled end time in EST in the format YY-MM-DD-HH-MM")
    notes: Optional[str] = Field(default=None, description="Additional notes")


CourseDateStatus = Union[
    CourseDateStatusHeld,
    CourseDateStatusCanceled,
    CourseDateStatusRescheduled,
]


class Course(BaseModel):
    course_id: str = Field(description="Unique identifier for the course")
    department: str = Field(description="Department offering the course")
    course_code: str = Field(description="Course code in the format DEPT-NNNN")
    title: str = Field(description="Course title")
    credits: int = Field(description="Number of credits")
    term: str = Field(description="Term in the format YYYY-TERM, e.g. 2025-SPRING")
    instructor: Instructor = Field(description="Instructor information")
    location: Location = Field(description="Course location")
    scheduled_first_meeting_est: str = Field(description="First scheduled meeting in EST in the format YY-MM-DD-HH-MM")
    scheduled_last_meeting_est: str = Field(description="Last scheduled meeting in EST in the format YY-MM-DD-HH-MM")
    meeting_pattern: MeetingPattern = Field(description="Weekly meeting pattern")
    dates: Dict[str, CourseDateStatus] = Field(description="Per-date class details keyed by date (YY-MM-DD)")
    capacity: int = Field(description="Maximum enrollment capacity")
    waitlist_capacity: int = Field(description="Maximum waitlist capacity")
    current_enrollment: int = Field(description="Current number of enrolled students")
    current_waitlist: int = Field(description="Current number of students on the waitlist")


# Student-related models
class StudentPaymentMethodExtraInfo(BaseModel):
    brand: Optional[str] = Field(default=None, description="Payment card brand, if applicable")
    last_four: Optional[str] = Field(default=None, description="Last four digits of the card, if applicable")


class StudentPaymentMethod(BaseModel):
    source: str = Field(description="Type/source of the payment method (e.g., credit_card, bank_account)")
    payment_method_id: str = Field(description="Unique identifier for the payment method")
    extra_info: StudentPaymentMethodExtraInfo = Field(description="Additional information for the payment method")


class SavedContact(BaseModel):
    first_name: str = Field(description="Contact's first name")
    last_name: str = Field(description="Contact's last name")
    relationship: str = Field(description="Relationship to the student")
    phone: str = Field(description="Contact phone number")
    date_of_birth: str = Field(
        description="Contact's date of birth in the format YY-MM-DD",
        alias="date-of-birth",
    )


class Program(BaseModel):
    school: str = Field(description="School or college name")
    degree: str = Field(description="Degree type (e.g., BA, BS, MS, PhD, etc.)")
    major: str = Field(description="Declared major")
    minor: Optional[str] = Field(default=None, description="Declared minor, if any")


class Student(BaseModel):
    student_id: str = Field(description="Unique identifier for the student")
    name: Name = Field(description="Student's full name")
    address: StudentAddress = Field(description="Student's address")
    email: str = Field(description="Student's email address")
    dob: str = Field(description="Student's date of birth in the format YY-MM-DD")
    payment_methods: Dict[str, StudentPaymentMethod] = Field(description="Saved payment methods keyed by payment method ID")
    saved_contacts: List[SavedContact] = Field(description="Saved contacts for the student")
    program: Program = Field(description="Academic program details")
    membership: Optional[str] = Field(default=None, description="Student memberships (e.g., honors, student_union, athletics)")
    registrations: List[str] = Field(description="List of registration IDs for the student")


# Registration-related models
class RegistrationCourse(BaseModel):
    course_id: str = Field(description="Course ID")
    course_code: str = Field(description="Course code in the format DEPT-NNNN")
    section: str = Field(description="Section identifier")
    title: str = Field(description="Course title")
    credits: int = Field(description="Number of credits for the course")
    grading_option: GradingOption = Field(description="Grading option")
    tuition: int = Field(description="Tuition amount for this course in dollars")


class Advisor(BaseModel):
    first_name: str = Field(description="Advisor's first name")
    last_name: str = Field(description="Advisor's last name")
    email: str = Field(description="Advisor's email address")


class FinancialAid(BaseModel):
    scholarships: int = Field(description="Total scholarships amount in dollars")
    grants: int = Field(description="Total grants amount in dollars")
    loans: int = Field(description="Total loans amount in dollars")


class Registration(BaseModel):
    registration_id: str = Field(description="Unique identifier for the registration")
    student_id: str = Field(description="ID of the student")
    campus: str = Field(description="Campus for this registration")
    program: str = Field(description="Program name for this registration")
    study_level: StudyLevel = Field(description="Study level")
    term: str = Field(description="Term in the format YYYY-TERM")
    status: EnrollmentStatus = Field(description="Enrollment status")
    courses: List[RegistrationCourse] = Field(description="Courses included in the registration")
    advisors: List[Advisor] = Field(description="Advisors assigned to the student")
    payment_history: List[Payment] = Field(description="History of payments for this registration")
    created_at: str = Field(description="Creation date in the format YY-MM-DD")
    total_credits: int = Field(description="Total credits registered")
    overload_credits: int = Field(description="Overload credits above the standard load")
    financial_aid: FinancialAid = Field(description="Financial aid breakdown")
    health_insurance: YesNo = Field(description="Whether the student has health insurance through the school")


# Database
class SchoolDB(DB):
    """Database of all courses, students, and registrations."""

    courses: Dict[str, Course] = Field(description="Dictionary of all courses indexed by course ID")
    students: Dict[str, Student] = Field(description="Dictionary of all students indexed by student ID")
    registrations: Dict[str, Registration] = Field(description="Dictionary of all registrations indexed by registration ID")

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_courses = len(self.courses)
        num_course_dates = sum(len(c.dates) for c in self.courses.values())
        num_students = len(self.students)
        num_registrations = len(self.registrations)
        return {
            "num_courses": num_courses,
            "num_course_dates": num_course_dates,
            "num_students": num_students,
            "num_registrations": num_registrations,
        }


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
