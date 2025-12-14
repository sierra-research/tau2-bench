"""
List All Product Types Command for FastWorkflow.

This command wraps tau2-bench's list_all_product_types tool.
"""

from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """List all available product types in the retail database."""
    
    class Input(BaseModel):
        # No parameters needed for this tool
        trigger: str = Field(
            default="list",
            description="Trigger to list product types",
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        product_types: str = Field(
            description="List of all available product types",
            json_schema_extra={
                "used_by": ["get_product_details"]
            }
        )

    plain_utterances: List[str] = [
        "List all product types",
        "Show me all available product categories",
        "What product types do you have?",
        "Display all product types in the system",
        "Get list of all product categories",
        "What kinds of products are available?",
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        import os
        command_name = os.path.splitext(os.path.basename(__file__))[0]
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(
            utterances_obj.plain_utterances, command_name
        )


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
                CommandResponse(response=f"Available product types:\n{output.product_types}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """List all product types using tau2-bench's list_all_product_types tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            # Returns List[str]
            product_types_list = tools.list_all_product_types()
            
            # Format as readable string
            product_types_str = "\n".join(f"- {pt}" for pt in product_types_list)
            
            return Signature.Output(product_types=product_types_str)
            
        except Exception as e:
            return Signature.Output(product_types=f"Error listing product types: {str(e)}")
