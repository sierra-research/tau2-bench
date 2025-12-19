from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """List all product types"""
    """Metadata and parameter definitions for `list_all_product_types`."""

    class Input(BaseModel):
        """No parameters expected for this command."""

    class Output(BaseModel):
        status: str = Field(
            description="List of product type and product id tuples, or a JSON string representation of that list.",
            json_schema_extra={
                "used_by": ["get_product_details"]
            }
        )

    # ---------------------------------------------------------------------
    # Utterances
    # ---------------------------------------------------------------------

    plain_utterances: List[str] = [
        "What kind of products do you have in the store?",
        "Can you show me everything you carry?",
        "I'm curious about all the categories you offer.",
        "I'd like to browse your full product range.",
        "What are the different types of items you sell?",
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        from fastworkflow.train.generate_synthetic import generate_diverse_utterances

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
                CommandResponse(response=f"Available product types:\n{output.status}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """List all product types using tau2-bench's list_all_product_types tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            product_types_json = tools.list_all_product_types()
            
            import json
            product_dict = json.loads(product_types_json)
            product_types_str = "\n".join(f"- {name}: {pid}" for name, pid in product_dict.items())
            
            return Signature.Output(status=product_types_str)
            
        except Exception as e:
            return Signature.Output(status=f"Error listing product types: {str(e)}")
