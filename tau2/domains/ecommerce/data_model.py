from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.environment.db import DB


# -----------------------------
# Catalogue models
# -----------------------------
class OfferingAttributes(BaseModel):
    """Attributes describing a specific offering (SKU) in the catalogue"""

    hue: str = Field(description="Color or hue of the item")
    dimension: str = Field(description="Size or dimensions of the item")
    fabrication: str = Field(description="Material or fabrication details")
    pattern: str = Field(description="Pattern or style")


class Offering(BaseModel):
    """Represents a specific unit (SKU) offering"""

    unit_sku: str = Field(description="Unique identifier for the unit SKU")
    attributes: OfferingAttributes = Field(
        description="Attributes of the offering (e.g., hue, dimension, fabrication, pattern)"
    )
    in_stock: bool = Field(description="Whether this unit is currently in stock")
    unit_price: float = Field(description="Price for a single unit of this offering")


class CatalogueGroup(BaseModel):
    """Represents a group in the catalogue with its offerings"""

    title: str = Field(description="Title/name of the catalogue group")
    group_ref: str = Field(description="Unique reference for the catalogue group")
    offerings: Dict[str, Offering] = Field(
        description="Dictionary of offerings indexed by unit_sku"
    )


# -----------------------------
# Account models
# -----------------------------
class PersonName(BaseModel):
    """Represents a person's name"""

    given: str = Field(description="Given (first) name")
    family: str = Field(description="Family (last) name")


class Location(BaseModel):
    """Represents a physical mailing address"""

    line1: str = Field(description="Primary address line")
    line2: str = Field(description="Secondary address line")
    municipality: str = Field(description="City or municipality")
    nation: str = Field(description="Country")
    region: str = Field(description="State or region")
    postal_code: str = Field(description="Postal or ZIP code")


class FundingSourceMeta(BaseModel):
    """Additional metadata for a funding source"""

    issuer: str = Field(description="Issuer or provider of the instrument (e.g., bank, network)")
    last_digits: str = Field(description="Last digits of the instrument for reference")


class FundingSource(BaseModel):
    """Represents a funding instrument associated with an account"""

    origin: str = Field(description="Type/origin of the instrument (e.g., credit_card, paypal, gift_card)")
    instrument_id: str = Field(description="Unique identifier for the funding instrument")
    meta: FundingSourceMeta = Field(description="Metadata describing the instrument")


class Account(BaseModel):
    """Represents a customer account, including profile, address, funding sources, and purchases"""

    account_key: str = Field(description="Unique key identifying the account")
    person: PersonName = Field(description="Person's name information")
    location: Location = Field(description="Account's primary address")
    contact_email: str = Field(description="Primary contact email for the account")
    funding_sources: Dict[str, FundingSource] = Field(
        description="Dictionary of funding sources indexed by instrument_id"
    )
    purchases: List[str] = Field(
        description="List of sale references (sale_ref) associated with this account"
    )


# -----------------------------
# Sales models
# -----------------------------
class SaleLineAttributes(BaseModel):
    """Attributes describing a line item within a sale"""

    volume: str = Field(description="Volume or size attribute")
    composition: str = Field(description="Composition or material details")
    tint: str = Field(description="Tint or color variation")


class SaleLine(BaseModel):
    """Represents an individual line item in a sale"""

    label: str = Field(description="Display label for the line item (e.g., product name)")
    catalog_ref: str = Field(description="Reference to the catalogue group")
    unit_sku: str = Field(description="SKU for the specific unit purchased")
    unit_price: float = Field(description="Price per unit at time of purchase")
    attributes: SaleLineAttributes = Field(description="Attributes for this line item")


class Shipment(BaseModel):
    """Represents a shipment tied to a sale"""

    parcel_codes: List[str] = Field(description="Tracking or parcel codes for the shipment")
    sku_list: List[str] = Field(description="List of SKUs included in this shipment")


SaleEntryKind = Literal["payment", "refund"]


class LedgerEntry(BaseModel):
    """Represents a payment or refund entry in the sale ledger"""

    entry_kind: SaleEntryKind = Field(description="Type of ledger entry (payment or refund)")
    value: float = Field(description="Monetary value of the entry")
    instrument_id: str = Field(description="Funding instrument used for this entry")


class Sale(BaseModel):
    """Represents a sale/order with delivery, line items, shipments, and financial ledger"""

    sale_ref: str = Field(description="Unique reference for the sale")
    account_key: str = Field(description="Account key associated with the sale")
    delivery: Location = Field(description="Delivery address for the sale")
    lines: List[SaleLine] = Field(description="Line items included in the sale")
    state: str = Field(description="Current state/status of the sale")
    shipments: List[Shipment] = Field(description="Shipments associated with the sale")
    ledger: List[LedgerEntry] = Field(description="Payment/refund entries for the sale")


# -----------------------------
# E-commerce database
# -----------------------------
class ECommerceDB(DB):
    """Database containing all e-commerce related data: catalogue, accounts, and sales"""

    catalogue: Dict[str, CatalogueGroup] = Field(
        description="Dictionary of catalogue groups indexed by group_ref"
    )
    accounts: Dict[str, Account] = Field(
        description="Dictionary of accounts indexed by account_key"
    )
    sales: Dict[str, Sale] = Field(
        description="Dictionary of sales indexed by sale_ref"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get basic statistics of the database."""
        num_groups = len(self.catalogue)
        num_accounts = len(self.accounts)
        num_sales = len(self.sales)
        total_num_offerings = sum(len(group.offerings) for group in self.catalogue.values())
        return {
            "num_groups": num_groups,
            "num_accounts": num_accounts,
            "num_sales": num_sales,
            "total_num_offerings": total_num_offerings,
        }
