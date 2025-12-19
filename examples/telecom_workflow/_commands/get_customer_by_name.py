from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH


class Signature:
    """Find customer by full name and date of birth"""
    
    class Input(BaseModel):
        full_name: str = Field(
            default="NOT_FOUND",
            description="The full name of the customer to search for",
            examples=["John Smith", "Jane Doe", "Robert Johnson"],
        )
        
        dob: str = Field(
            default="NOT_FOUND",
            description="Date of birth for verification in YYYY-MM-DD format",
            pattern=r"^(NOT_FOUND|\d{4}-\d{2}-\d{2})$",
            examples=["1990-01-15", "1985-12-31", "2000-06-20"],
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        customer_list: str = Field(
            description="List of matching customer objects with full details. May return multiple matches if names are similar.",
            json_schema_extra={
                "used_by": ["get_customer_by_id"]
            }
        )
    
    plain_utterances: List[str] = [
        "Can you find the customer named John Smith born on 1990-05-15?",
        "I need to look up a customer by name: Jane Doe, DOB 1985-11-22",
        "Search for customer Robert Johnson with birthday 1975-03-10",
        "Find account for Mary Williams, date of birth 1992-08-05",
        "Look up customer Michael Brown born 1988-12-30",
        "I'm trying to find Sarah Davis, her DOB is 1995-04-17",
        "Can you search for customer David Miller with birthdate 1980-09-25?",
        "Find the account for Jennifer Garcia born on 1993-07-14",
        "Look up customer Christopher Martinez, DOB: 1987-02-28",
        "Search for customer Elizabeth Rodriguez with birthday 1991-06-09",
    ]
    
    template_utterances: List[str] = []
    
    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"Found customers:\n{output.customer_list}")
            ]
        )
    
    def _process_command(
        self,
        workflow: Workflow,
        input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_customer_by_name command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            customers = tools.get_customer_by_name(
                full_name=input.full_name,
                dob=input.dob
            )
            
            if not customers:
                return Signature.Output(customer_list="No customers found matching the criteria")
            
            import json
            customers_json = json.dumps(
                [customer.model_dump() for customer in customers],
                indent=2,
                default=str
            )
            
            return Signature.Output(customer_list=customers_json)
            
        except ValueError as e:
            return Signature.Output(customer_list=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(customer_list=f"Unexpected error: {str(e)}")