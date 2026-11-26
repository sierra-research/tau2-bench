from typing import Optional
from pydantic import BaseModel, Field
from tau2.environment.db import DB


class Person(BaseModel):
    id: str = Field(description="Unique identifier for the person")
    first_name: str = Field(description="Person's first name")
    last_name: str = Field(description="Person's last name")
    email: Optional[str] = Field(None, description="Person's primary email")
    phone: Optional[str] = Field(None, description="Person's primary phone")
    job_title: Optional[str] = Field(None, description="Person's job title")
    company_id: Optional[str] = Field(None, description="Associated company ID")
    lead_score: Optional[int] = Field(None, description="Lead score (0-100)")
    relationship_stage: Optional[str] = Field(None, description="Current relationship stage")
    trust_level: Optional[int] = Field(None, description="Trust level (0-100)")
    commitment_level: Optional[int] = Field(None, description="Commitment level (0-100)")


class Company(BaseModel):
    id: str = Field(description="Unique identifier for the company")
    name: str = Field(description="Company name")
    industry: Optional[str] = Field(None, description="Company industry")
    annual_revenue: Optional[float] = Field(None, description="Annual revenue")
    employee_count: Optional[int] = Field(None, description="Number of employees")


class Opportunity(BaseModel):
    id: str = Field(description="Unique identifier for the opportunity")
    name: str = Field(description="Opportunity name")
    person_id: str = Field(description="Associated person ID")
    company_id: Optional[str] = Field(None, description="Associated company ID")
    stage: str = Field(description="Current stage in sales pipeline")
    value: Optional[float] = Field(None, description="Opportunity value")
    probability: Optional[float] = Field(None, description="Success probability 0-1")
    expected_close_date: Optional[str] = Field(None, description="Expected close date")


class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task")
    subject: str = Field(description="Task subject/description")
    person_id: Optional[str] = Field(None, description="Associated person ID")
    company_id: Optional[str] = Field(None, description="Associated company ID")
    status: str = Field(description="Task status (open, completed, cancelled)")
    priority: Optional[str] = Field(None, description="Task priority")


class CRMDB(DB):
    """Database for CRM domain data."""
    
    people: dict[str, Person] = Field(description="Dictionary of people indexed by ID")
    companies: dict[str, Company] = Field(description="Dictionary of companies indexed by ID")
    opportunities: dict[str, Opportunity] = Field(description="Dictionary of opportunities indexed by ID")
    tasks: dict[str, Task] = Field(description="Dictionary of tasks indexed by ID")
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        return {
            "num_people": len(self.people),
            "num_companies": len(self.companies),
            "num_opportunities": len(self.opportunities),
            "num_tasks": len(self.tasks),
        }
