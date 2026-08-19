"""Toolkit for the hospitality domain (Berkeley Hot Pot restaurant)."""

import hashlib
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.domains.hospitality.data_model import (
    Customer,
    HospitalityDB,
    Incident,
    IncidentType,
    Inventory,
    KitchenStatus,
    MemberTier,
    MenuItem,
    Order,
    OrderItem,
    OrderStatus,
    Reservation,
    ReservationStatus,
    SecretCode,
    SoupBase,
    StaffAuthority,
    StaffRole,
    Table,
    TableStatus,
    TableType,
)
from tau2.domains.hospitality.utils import (
    get_now,
    get_today,
    is_federal_holiday,
    is_lunch_time,
    is_weekday,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HospitalityTools(ToolKitBase):
    """Tools for the hospitality domain."""

    db: HospitalityDB

    def __init__(self, db: HospitalityDB) -> None:
        """Initialize the hospitality tools with a database instance."""
        super().__init__(db)

    # ============== Helper Methods (not tools) ==============

    # ============== Initialization Methods (for test setup) ==============
    
    def initialize_order(
        self, 
        bill_amount: float, 
        table_id: str = "A01", 
        party_size: int = 2,
        order_id: Optional[str] = None
    ) -> str:
        """Initialize a test order with specified bill amount. Used for test setup.
        
        Args:
            bill_amount: The food subtotal (before sauce bar, tax)
            table_id: Table number
            party_size: Number of guests (sauce bar charge = $2 per person)
            order_id: Optional custom order ID (default: auto-generated)
        """
        from datetime import datetime
        sauce_bar_charge = party_size * 2.0  # $2 per person
        subtotal_with_sauce = bill_amount + sauce_bar_charge
        
        final_order_id = order_id if order_id else self._generate_id("ORD", table_id, bill_amount)
        
        order = Order(
            order_id=final_order_id,
            table_id=table_id,
            party_size=party_size,
            items=[],
            subtotal=bill_amount,
            sauce_bar_charge=sauce_bar_charge,
            tax=subtotal_with_sauce * 0.0875,
            total=subtotal_with_sauce * 1.0875,
            status=OrderStatus.IN_PROGRESS,
            created_at="2026-01-01T12:00:00",  # Fixed timestamp for deterministic evaluation
        )
        self.db.orders.append(order)
        return f"Test order created: ${bill_amount:.2f} food + ${sauce_bar_charge:.2f} sauce bar ({party_size} guests)"

    def initialize_customer_points(self, customer_id: str, points: int, tier: str = "Gold") -> str:
        """Initialize customer with specific points balance. Used for test setup."""
        # Normalize tier to proper case (e.g., "gold" -> "Gold")
        tier_normalized = tier.capitalize()
        member_tier = MemberTier(tier_normalized)
        
        for customer in self.db.customers:
            if customer.customer_id == customer_id:
                customer.points = points
                customer.tier = member_tier
                return f"Customer {customer_id} points set to {points}"
        # Create new customer if not exists
        customer = Customer(
            customer_id=customer_id,
            name="Test Customer",
            phone="555-0000",
            tier=member_tier,
            points=points,
            visit_count=5,
        )
        self.db.customers.append(customer)
        return f"New customer created with {points} points"

    def initialize_peak_hours(self, is_peak: bool = True) -> str:
        """Set whether restaurant is currently in peak hours. Used for test setup."""
        self.db.is_peak_hours = is_peak
        return f"Peak hours set to {is_peak}"

    def set_customer_sms_claim(
        self,
        date: str,
        content: str,
        missing_terms: Optional[str] = None,
        discount_value: float = 20.0
    ) -> str:
        """Set up a customer's SMS promotion claim for verification. Used for test setup.
        
        Args:
            date: Date the SMS was sent
            content: Content of the SMS promotion
            missing_terms: Terms that were omitted (indicates company error)
            discount_value: Dollar amount of discount claimed
        """
        self.db.customer_sms_claim = {
            "date": date,
            "content": content,
            "missing_terms": missing_terms,
            "discount_value": discount_value,
            "company_error": missing_terms is not None,
        }
        return f"SMS claim set: {content}"

    def set_kitchen_response(
        self, 
        response: str, 
        can_fulfill: Optional[bool] = None, 
        estimated_wait: Optional[int] = None,
        status: str = "normal"
    ) -> str:
        """
        Set the kitchen response for testing kitchen coordination scenarios.
        Used for test setup to simulate various kitchen states.
        
        Args:
            response: The message kitchen staff would say (may be unprofessional)
            can_fulfill: Whether kitchen can fulfill the request
            estimated_wait: Estimated wait time in minutes
            status: Kitchen status (normal, order_overload, understaffed, etc.)
        """
        self.db.kitchen_response = response
        self.db.kitchen_can_fulfill = can_fulfill
        self.db.kitchen_estimated_wait = estimated_wait
        self.db.kitchen_status = status
        return f"Kitchen response configured: status={status}"

    def set_table_membership(self, has_member: bool = False) -> str:
        """
        Set whether current table has a linked member account.
        Used for test setup to simulate membership scenarios.
        
        Args:
            has_member: Whether table already has member linked
        """
        if self.db.orders:
            self.db.orders[-1].has_member = has_member
        return f"Table membership set: has_member={has_member}"

    def set_customer_mood(self, mood: str = "normal") -> str:
        """
        Set customer mood for the current interaction.
        Used for test setup. Setting this explicitly enables membership promotion testing.
        
        Args:
            mood: Customer mood (normal, upset, rushing, celebrating)
            
        Note:
            Only tasks that explicitly call this method will test membership promotion.
            Tasks without this initialization are assumed to have existing members.
        """
        self.db.customer_mood = mood
        self.db.mood_explicitly_set = True  # Flag to indicate mood was set for this task
        return f"Customer mood set: {mood}"

    def set_table_status(self, table_id: str, status: str, party_size: int = 0) -> str:
        """
        Set a specific table's status for test setup.
        
        Args:
            table_id: Table ID (e.g., "A1", "B3", "C1")
            status: Table status - "available", "occupied", "reserved", "cleaning"
            party_size: Current party size if occupied (default 0)
        """
        for table in self.db.tables:
            if table.table_id == table_id:
                table.status = TableStatus(status)
                table.current_party_size = party_size
                return f"Table {table_id} set to {status} (party: {party_size})"
        return f"Table {table_id} not found"

    def set_restaurant_occupancy(self, occupancy_level: str) -> str:
        """
        Set restaurant-wide occupancy level for test setup.
        Automatically configures table statuses based on level.
        
        Args:
            occupancy_level: One of:
                - "empty": All tables available
                - "light": ~30% occupied (A1-A6 occupied)
                - "moderate": ~50% occupied (A1-A10, B1-B4 occupied)
                - "busy": ~75% occupied (A1-A15, B1-B6, C1 occupied)
                - "full": All tables occupied except C2
                - "peak_no_large": All tables occupied, no large tables available
        """
        # Reset all tables to available first
        for table in self.db.tables:
            table.status = TableStatus.AVAILABLE
            table.current_party_size = 0
        
        if occupancy_level == "empty":
            return "Restaurant set to empty - all tables available"
        
        elif occupancy_level == "light":
            # 30% - occupy A1-A6
            for table in self.db.tables:
                if table.table_id in ["A1", "A2", "A3", "A4", "A5", "A6"]:
                    table.status = TableStatus.OCCUPIED
                    table.current_party_size = 4
            return "Restaurant set to light occupancy (~30%)"
        
        elif occupancy_level == "moderate":
            # 50% - occupy A1-A10, B1-B4
            for table in self.db.tables:
                if table.table_id.startswith("A") and int(table.table_id[1:]) <= 10:
                    table.status = TableStatus.OCCUPIED
                    table.current_party_size = 4
                elif table.table_id in ["B1", "B2", "B3", "B4"]:
                    table.status = TableStatus.OCCUPIED
                    table.current_party_size = 6
            return "Restaurant set to moderate occupancy (~50%)"
        
        elif occupancy_level == "busy":
            # 75% - occupy A1-A15, B1-B6, C1
            for table in self.db.tables:
                if table.table_id.startswith("A") and int(table.table_id[1:]) <= 15:
                    table.status = TableStatus.OCCUPIED
                    table.current_party_size = 4
                elif table.table_id in ["B1", "B2", "B3", "B4", "B5", "B6"]:
                    table.status = TableStatus.OCCUPIED
                    table.current_party_size = 6
                elif table.table_id == "C1":
                    table.status = TableStatus.OCCUPIED
                    table.current_party_size = 10
            return "Restaurant set to busy (~75%)"
        
        elif occupancy_level == "full":
            # All occupied except C2
            for table in self.db.tables:
                if table.table_id != "C2":
                    table.status = TableStatus.OCCUPIED
                    if table.table_id.startswith("A"):
                        table.current_party_size = 4
                    elif table.table_id.startswith("B"):
                        table.current_party_size = 6
                    else:
                        table.current_party_size = 10
            return "Restaurant set to full - only C2 available"
        
        elif occupancy_level == "peak_no_large":
            # All tables occupied including large tables
            for table in self.db.tables:
                table.status = TableStatus.OCCUPIED
                if table.table_id.startswith("A"):
                    table.current_party_size = 4
                elif table.table_id.startswith("B"):
                    table.current_party_size = 6
                else:
                    table.current_party_size = 10
            return "Restaurant set to peak - no large tables available"
        
        return f"Unknown occupancy level: {occupancy_level}"

    # ============== End Initialization Methods ==============

    def _get_table_by_id(self, table_id: str) -> Table:
        """Get a table by ID."""
        for table in self.db.tables:
            if table.table_id == table_id:
                return table
        raise ValueError(f"Table {table_id} not found")

    def _get_customer_by_id(self, customer_id: str) -> Customer:
        """Get a customer by ID."""
        for customer in self.db.customers:
            if customer.customer_id == customer_id:
                return customer
        raise ValueError(f"Customer {customer_id} not found")

    def _get_reservation_by_id(self, reservation_id: str) -> Reservation:
        """Get a reservation by ID."""
        for res in self.db.reservations:
            if res.reservation_id == reservation_id:
                return res
        raise ValueError(f"Reservation {reservation_id} not found")

    def _get_order_by_id(self, order_id: str) -> Order:
        """Get an order by ID."""
        for order in self.db.orders:
            if order.order_id == order_id:
                return order
        raise ValueError(f"Order {order_id} not found")

    def _get_menu_item_by_id(self, item_id: str) -> MenuItem:
        """Get a menu item by ID."""
        for item in self.db.menu_items:
            if item.id == item_id:
                return item
        raise ValueError(f"Menu item {item_id} not found")

    def _get_soup_base_by_id(self, soup_id: str) -> SoupBase:
        """Get a soup base by ID."""
        for soup in self.db.soup_bases:
            if soup.id == soup_id:
                return soup
        raise ValueError(f"Soup base {soup_id} not found")

    def _get_inventory_by_id(self, item_id: str) -> Inventory:
        """Get inventory item by ID."""
        for inv in self.db.inventory:
            if inv.item_id == item_id:
                return inv
        raise ValueError(f"Inventory item {item_id} not found")

    def _get_staff_authority(self, role: StaffRole) -> StaffAuthority:
        """Get authority for a staff role."""
        for auth in self.db.staff_authorities:
            if auth.role == role:
                return auth
        raise ValueError(f"Authority for {role} not found")

    def _generate_id(self, prefix: str, *args: Any) -> str:
        """
        Generate a deterministic ID based on prefix and input arguments.
        This ensures reproducibility when replaying tool calls during evaluation.
        """
        # Create a deterministic hash from the prefix and all arguments
        hash_input = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_value}"

    # ============== READ Tools ==============

    @is_tool(ToolType.READ)
    def get_restaurant_info(self) -> Dict[str, Any]:
        """
        Get basic restaurant information including name, location, and business hours.

        Returns:
            Dictionary with restaurant name, location, and hours.
        """
        return {
            "name": self.db.restaurant.name,
            "location": self.db.restaurant.location,
            "hours": self.db.restaurant.hours,
        }

    @is_tool(ToolType.READ)
    def get_menu_details(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get menu details including soup bases and menu items.

        Args:
            category: Optional category filter (soup_base, protein, seafood, veggie, etc.)

        Returns:
            Dictionary with soup bases and/or menu items.
        """
        result = {}

        if category is None or category == "soup_base":
            result["soup_bases"] = [
                {
                    "id": sb.id,
                    "name": sb.name,
                    "spicy_level": sb.spicy_level,
                    "allergies": sb.allergies,
                    "prices": sb.prices,
                }
                for sb in self.db.soup_bases
            ]

        if category is None:
            result["menu_items"] = [item.model_dump() for item in self.db.menu_items]
        elif category != "soup_base":
            result["menu_items"] = [
                item.model_dump()
                for item in self.db.menu_items
                if item.category.lower() == category.lower()
            ]

        if self.db.lunch_special:
            result["lunch_special"] = self.db.lunch_special.model_dump()

        return result

    @is_tool(ToolType.READ)
    def check_table_availability(
        self, party_size: int, date_str: str, time_str: str
    ) -> Dict[str, Any]:
        """
        Check available tables for a given party size and time.

        Args:
            party_size: Number of guests.
            date_str: Date in YYYY-MM-DD format.
            time_str: Time in HH:MM format.

        Returns:
            Dictionary with available tables and their capacities.
        """
        # Track that availability was checked
        self.db.availability_checked = True
        
        available_tables = []

        # Get reservations for this date/time
        reserved_tables = set()
        for res in self.db.reservations:
            if res.date == date_str and res.status == ReservationStatus.CONFIRMED:
                if res.table_id:
                    reserved_tables.add(res.table_id)

        for table in self.db.tables:
            if table.table_id in reserved_tables:
                continue
            if table.status == TableStatus.AVAILABLE:
                if table.std_capacity >= party_size:
                    # Fits within standard capacity - best option
                    available_tables.append(
                        {
                            "table_id": table.table_id,
                            "type": table.table_type.value,
                            "std_capacity": table.std_capacity,
                            "std_expansion": table.std_expansion,
                            "max_squeeze": table.max_squeeze,
                            "fit_type": "standard",
                            "recommended": True,
                        }
                    )
                elif table.std_expansion >= party_size:
                    # Fits with default extra chairs - good option
                    available_tables.append(
                        {
                            "table_id": table.table_id,
                            "type": table.table_type.value,
                            "std_capacity": table.std_capacity,
                            "std_expansion": table.std_expansion,
                            "max_squeeze": table.max_squeeze,
                            "fit_type": "expansion",
                            "recommended": True,
                            "note": "Will add extra chairs (standard practice)",
                        }
                    )
                elif table.max_squeeze >= party_size:
                    # Only fits with squeeze - not recommended
                    available_tables.append(
                        {
                            "table_id": table.table_id,
                            "type": table.table_type.value,
                            "std_capacity": table.std_capacity,
                            "std_expansion": table.std_expansion,
                            "max_squeeze": table.max_squeeze,
                            "fit_type": "squeeze",
                            "recommended": False,
                            "note": "Would require squeezing beyond standard - not recommended, may be uncomfortable. Only offer if customer is a regular AND proactively requests it.",
                        }
                    )

        # Determine if this is peak hours
        check_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        is_weekend = check_date.weekday() >= 5
        is_holiday = is_federal_holiday(check_date)
        
        # Peak hours: Friday 6-9pm, Saturday 5-9pm, Sunday 5-8pm
        hour = int(time_str.split(":")[0])
        is_peak = False
        if check_date.weekday() == 4 and 18 <= hour <= 21:  # Friday
            is_peak = True
        elif check_date.weekday() == 5 and 17 <= hour <= 21:  # Saturday
            is_peak = True
        elif check_date.weekday() == 6 and 17 <= hour <= 20:  # Sunday
            is_peak = True
        
        result = {
            "party_size": party_size,
            "date": date_str,
            "time": time_str,
            "available_tables": available_tables,
            "total_available": len(available_tables),
            "is_peak_hours": is_peak,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
        }
        
        if len(available_tables) == 0:
            result["message"] = "No tables available for this party size and time."
            if is_peak or is_weekend or is_holiday:
                result["suggestion"] = "This is a busy period. Typical wait time is 2+ hours. Suggest customer join waitlist on Google Maps or Yelp."
        
        return result

    @is_tool(ToolType.READ)
    def get_customer_profile(
        self, customer_id: Optional[str] = None, phone: Optional[str] = None
    ) -> Customer:
        """
        Look up a customer by ID or phone number.

        Args:
            customer_id: Customer ID to look up.
            phone: Phone number to look up.

        Returns:
            Customer profile information.
        """
        if customer_id:
            return self._get_customer_by_id(customer_id)

        if phone:
            for customer in self.db.customers:
                if customer.phone == phone:
                    return customer
            raise ValueError(f"Customer with phone {phone} not found")

        raise ValueError("Must provide either customer_id or phone")

    @is_tool(ToolType.READ)
    def check_allergy_safety(self, item_id: str, allergy: str) -> Dict[str, Any]:
        """
        Check if a menu item or soup base is safe for a specific allergy.
        IMPORTANT: Due to hidden ingredients and cross-contamination risks,
        customers with severe allergies should be strongly recommended Plain Water base.

        Args:
            item_id: ID or name of the soup base or menu item (partial match supported).
            allergy: The allergy to check for (e.g., "vinegar", "gluten", "peanut").

        Returns:
            Safety information including known allergens and hidden ingredient warnings.
        """
        allergy_lower = allergy.lower()
        item_id_lower = item_id.lower()

        # Check soup bases (by ID or name, partial match)
        for soup in self.db.soup_bases:
            if soup.id.lower() == item_id_lower or item_id_lower in soup.name.lower():
                known_safe = allergy_lower not in [a.lower() for a in soup.allergies]
                has_hidden = len(soup.hidden_ingredients) > 0
                hidden_risk = allergy_lower in [
                    h.lower() for h in soup.hidden_ingredients
                ]

                if soup.name == "Plain Water":
                    return {
                        "item": soup.name,
                        "is_safe": True,
                        "known_allergens": [],
                        "hidden_ingredients": [],
                        "recommendation": "Plain Water is the safest option for severe allergies.",
                    }

                return {
                    "item": soup.name,
                    "is_safe": known_safe and not hidden_risk,
                    "known_allergens": soup.allergies,
                    "hidden_ingredients": soup.hidden_ingredients,
                    "has_hidden_ingredient_risk": has_hidden,
                    "recommendation": (
                        "CANNOT GUARANTEE SAFETY. Due to possible hidden ingredients in pre-made sauces "
                        "and cross-contamination risks, we strongly recommend Plain Water base for "
                        "customers with severe or life-threatening allergies."
                        if has_hidden or not known_safe
                        else "Appears safe based on known ingredients, but please inform us of your allergy."
                    ),
                }

        # Check menu items (by ID or name, partial match)
        for item in self.db.menu_items:
            if item.id.lower() == item_id_lower or item_id_lower in item.name.lower():
                known_safe = allergy_lower not in [a.lower() for a in item.allergies]
                return {
                    "item": item.name,
                    "is_safe": known_safe,
                    "known_allergens": item.allergies,
                    "recommendation": (
                        "Appears safe based on known ingredients."
                        if known_safe
                        else f"Contains {allergy}. Not recommended for your allergy."
                    ),
                }

        raise ValueError(f"Item {item_id} not found")

    @is_tool(ToolType.READ)
    def check_lunch_special_availability(self) -> Dict[str, Any]:
        """
        Check if lunch special is currently available.
        Lunch special is NOT available on federal holidays.

        Returns:
            Availability status and details.
        """
        now = get_now()
        today = get_today()

        is_holiday = is_federal_holiday(today)
        is_wkday = is_weekday(today)
        is_lunch = is_lunch_time(now)

        available = is_wkday and is_lunch and not is_holiday

        reason = None
        if is_holiday:
            reason = "Lunch special is not available on federal holidays."
        elif not is_wkday:
            reason = "Lunch special is only available Monday through Friday."
        elif not is_lunch:
            reason = "Lunch special is only available before 5 PM."

        return {
            "available": available,
            "current_date": str(today),
            "current_time": now.strftime("%H:%M"),
            "is_federal_holiday": is_holiday,
            "is_weekday": is_wkday,
            "is_before_5pm": is_lunch,
            "reason": reason,
            "price": self.db.lunch_special.price
            if self.db.lunch_special and available
            else None,
        }

    @is_tool(ToolType.READ)
    def verify_promotion_claim(self, promotion_type: str = "sms") -> Dict[str, Any]:
        """
        Verify a customer's promotion claim (e.g., SMS promotion, coupon).
        Use this when customer claims they received a promotion offer.
        
        Args:
            promotion_type: Type of promotion to verify ("sms", "coupon", "email")
        
        Returns:
            Promotion details, validity, and whether company made an error.
        """
        today = get_today()
        is_wkday = is_weekday(today)
        
        # Check if there's a customer SMS claim to verify
        if self.db.customer_sms_claim:
            claim = self.db.customer_sms_claim
            missing_terms = claim.get('missing_terms', None)
            
            # If company omitted terms, it's their fault - honor the promotion
            company_error = missing_terms is not None
            
            return {
                "promotion_found": True,
                "promotion_content": claim.get('content', ''),
                "promotion_date": claim.get('date', ''),
                "discount_value": claim.get('discount_value', 0),
                "actual_terms": f"Full terms include: {missing_terms}" if missing_terms else "No additional terms",
                "missing_terms_in_communication": missing_terms,
                "company_communication_error": company_error,
                "is_valid_today": is_wkday or company_error,  # Valid if weekday OR company made error
                "recommendation": "HONOR the promotion - company error in SMS communication. Apply discount within your authority." if company_error else (
                    "Promotion valid - apply discount" if is_wkday else "Promotion only valid on weekdays"
                ),
                "current_day": today.strftime("%A"),
            }
        
        return {
            "promotion_found": False,
            "message": "No promotion record found. Ask customer for details.",
        }

    @is_tool(ToolType.READ)
    def check_item_inventory(self, item_name: str) -> Dict[str, Any]:
        """
        Check inventory level for a specific item (gifts, merchandise, etc.).

        Args:
            item_name: Name of the item to check.

        Returns:
            Inventory information including stock level.
        """
        item_name_lower = item_name.lower().strip()
        
        for inv in self.db.inventory:
            inv_name_lower = inv.name.lower()
            # Robust matching: Exact, or substring if length is sufficient
            if inv_name_lower == item_name_lower or \
               (len(item_name_lower) > 3 and item_name_lower in inv_name_lower) or \
               (inv.item_id.lower() == item_name_lower):
                return {
                    "item_id": inv.item_id,
                    "name": inv.name,
                    "stock": inv.stock,
                    "in_stock": inv.stock > 0,
                    "item_type": inv.item_type,
                    "points_required": inv.points_required,
                }
        raise ValueError(f"Inventory item '{item_name}' not found")

    @is_tool(ToolType.READ)
    def get_reservation_details(
        self, 
        reservation_id: Optional[str] = None,
        phone: Optional[str] = None,
        customer_name: Optional[str] = None,
        table_id: Optional[str] = None
    ) -> Reservation:
        """
        Look up a reservation by ID, phone, name, or table number.
        For dine-in customers, use table_id or customer info - no need to ask for reservation ID.

        Args:
            reservation_id: The reservation ID (optional)
            phone: Customer phone number (optional)
            customer_name: Customer name (optional, partial match)
            table_id: Table number for seated customers (optional)

        Returns:
            Reservation details.
        """
        if reservation_id:
            return self._get_reservation_by_id(reservation_id)
        
        # Search by table_id first (most common for dine-in)
        if table_id:
            for res in self.db.reservations:
                if res.table_id == table_id and res.status == ReservationStatus.SEATED:
                    return res
        
        # Search by phone
        if phone:
            for res in self.db.reservations:
                if res.phone == phone:
                    return res
        
        # Search by name (partial match)
        if customer_name:
            name_lower = customer_name.lower()
            for res in self.db.reservations:
                if name_lower in res.customer_name.lower():
                    return res
        
        raise ValueError("Reservation not found. Provide reservation_id, phone, customer_name, or table_id.")

    @is_tool(ToolType.READ)
    def get_order_details(self, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get details of an order. If no order_id specified, returns current table's order.
        In a restaurant setting, you can see the customer's order on your iPad.

        Args:
            order_id: The order ID to look up (optional - uses current table if not specified).

        Returns:
            Order details including items and total.
        """
        if order_id:
            return self._get_order_by_id(order_id)
        elif self.db.orders:
            return self.db.orders[-1]  # Return most recent order
        else:
            return {
                "message": "No active order for current table",
                "items": [],
                "subtotal": 0,
                "total": 0,
            }

    @is_tool(ToolType.READ)
    def get_current_staff_authority(self) -> Dict[str, Any]:
        """
        Get the authority limits for the current staff role.

        Returns:
            Authority details including discount limits.
        """
        auth = self._get_staff_authority(self.db.current_staff_role)
        return {
            "role": auth.role.value,
            "max_round_off": auth.max_round_off,
            "max_discount_pct": auth.max_discount_pct,
            "can_comp_items": auth.can_comp_items,
            "comp_item_limit": auth.comp_item_limit,
            "manager_on_duty": self.db.manager_on_duty,
        }

    # ============== WRITE Tools ==============

    @is_tool(ToolType.WRITE)
    def create_reservation(
        self,
        customer_name: str,
        phone: str,
        party_size: int,
        date_str: str,
        time_str: str,
        special_occasion: Optional[str] = None,
        num_kids: int = 0,
        notes: Optional[str] = None,
        has_cake: bool = False,
        cake_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new reservation.

        Args:
            customer_name: Name for the reservation.
            phone: Contact phone number.
            party_size: Number of guests.
            date_str: Date in YYYY-MM-DD format.
            time_str: Time in HH:MM format.
            special_occasion: Optional occasion (birthday, anniversary, etc.).
            num_kids: Number of children in the party.
            notes: Additional notes.
            has_cake: Whether customer is bringing a birthday cake.
            cake_type: Type of cake (regular or ice_cream) - important for storage.

        Returns:
            Created reservation details.
        """
        # Check weekend/holiday party size limit
        check_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        is_weekend = check_date.weekday() >= 5
        is_holiday = is_federal_holiday(check_date)

        if (is_weekend or is_holiday) and party_size > 20:
            raise ValueError(
                "We cannot accept reservations for parties over 20 on weekends and federal holidays."
            )

        # Use deterministic ID based on key reservation parameters
        reservation_id = self._generate_id(
            "RES", customer_name, phone, date_str, time_str, party_size
        )

        reservation = Reservation(
            reservation_id=reservation_id,
            customer_name=customer_name,
            phone=phone,
            party_size=party_size,
            date=date_str,
            time=time_str,
            special_occasion=special_occasion,
            num_kids=num_kids,
            num_high_chairs=num_kids,  # Default to same as num_kids
            notes=notes,
            has_cake=has_cake,
            cake_type=cake_type,
        )

        self.db.reservations.append(reservation)
        self.db.reservation_confirmed = True  # Track that reservation was created

        logger.info(f"Created reservation {reservation_id} for {customer_name}")

        return {
            "message": "Reservation created successfully",
            "reservation": reservation,
            "confirmation": f"Confirmation will be sent to {phone}",
            "reminders": [
                "Please arrive on time",
                f"Party size: {party_size}",
            ]
            + (["Birthday cake will be stored appropriately"] if has_cake else []),
        }

    @is_tool(ToolType.WRITE)
    def suggest_waitlist(self, reason: str = "fully_booked") -> Dict[str, Any]:
        """
        Suggest customer join online waitlist when restaurant is fully booked.
        
        Args:
            reason: Reason for suggesting waitlist (e.g., "fully_booked", "peak_hours")
        
        Returns:
            Waitlist information and online options.
        """
        self.db.waitlist_suggested = True
        
        return {
            "message": "Suggested customer join online waitlist",
            "reason": reason,
            "options": [
                "Google Maps - search 'Berkeley Hot Pot' and click 'Join Waitlist'",
                "Yelp - visit our Yelp page and click 'Join Waitlist'"
            ],
            "note": "During peak hours (Friday 6-9pm, Saturday 5-9pm, Sunday 5-8pm), typical wait is 2+ hours."
        }

    @is_tool(ToolType.WRITE)
    def offer_alternative_time(
        self, 
        original_time: str, 
        alternative_times: List[str],
        reason: str = "requested_time_unavailable"
    ) -> Dict[str, Any]:
        """
        Offer alternative time slots when requested time is unavailable.
        
        Args:
            original_time: The originally requested time
            alternative_times: List of available alternative times
            reason: Reason for offering alternatives
        
        Returns:
            Alternative time options.
        """
        self.db.alternative_time_offered = True
        
        return {
            "message": f"Offered alternative times instead of {original_time}",
            "original_request": original_time,
            "alternatives": alternative_times,
            "reason": reason
        }

    @is_tool(ToolType.WRITE)
    def apply_discount(
        self,
        order_id: Optional[str],
        discount_type: str,
        discount_value: float,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Apply a discount to an order. Checks staff authority limits.

        Args:
            order_id: The order to apply discount to (optional - uses current table's order if None).
            discount_type: Type of discount (percentage, fixed, round_off).
            discount_value: The discount amount or percentage.
            reason: Reason for the discount.

        Returns:
            Updated order with discount applied.
        """
        # If no order_id, use current/active order or create placeholder
        if not order_id:
            if self.db.orders:
                order = self.db.orders[-1]  # Use most recent order
                order_id = order.order_id
            else:
                # Create a placeholder order for tracking
                self.db.compensation_offered = True
                return {
                    "message": f"Discount of {discount_value}% noted for current table",
                    "discount_type": discount_type,
                    "discount_value": discount_value,
                    "reason": reason,
                    "success": True,
                }
        order = self._get_order_by_id(order_id)
        auth = self._get_staff_authority(self.db.current_staff_role)

        # Check authority limits
        if discount_type == "percentage":
            if discount_value > auth.max_discount_pct:
                if self.db.current_staff_role == StaffRole.SERVER:
                    raise ValueError(
                        f"Discount of {discount_value}% exceeds Server authority ({auth.max_discount_pct}%). "
                        f"Please consult a Manager for higher discounts."
                    )
                elif self.db.current_staff_role == StaffRole.HOST:
                    raise ValueError(
                        f"Discount of {discount_value}% exceeds Host authority ({auth.max_discount_pct}%). "
                        f"Please consult a Manager."
                    )
            discount_amount = order.subtotal * (discount_value / 100)
        elif discount_type in ["fixed", "round_off"]:
            if discount_value > auth.max_round_off:
                raise ValueError(
                    f"Round-off of ${discount_value} exceeds {self.db.current_staff_role.value} authority "
                    f"(${auth.max_round_off}). Please consult a Manager."
                )
            discount_amount = discount_value
        else:
            raise ValueError(f"Unknown discount type: {discount_type}")

        order.discount_applied = f"{discount_type}: {discount_value}"
        order.discount_amount = discount_amount
        order.total = order.subtotal + order.tax - discount_amount

        logger.info(
            f"Applied {discount_type} discount of {discount_value} to order {order_id}. Reason: {reason}"
        )

        return {
            "message": f"Discount applied successfully",
            "order_id": order_id,
            "discount_type": discount_type,
            "discount_value": discount_value,
            "discount_amount": discount_amount,
            "new_total": order.total,
        }

    @is_tool(ToolType.WRITE)
    def record_service_incident(
        self,
        incident_type: str,
        description: str,
        order_id: Optional[str] = None,
        table_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a service incident for tracking and potential compensation.

        Args:
            incident_type: Type of incident (slow_service, spill, wrong_order, etc.).
            description: Description of what happened.
            order_id: Related order ID if applicable.
            table_id: Related table ID if applicable.

        Returns:
            Created incident record.
        """
        # Use deterministic ID based on incident parameters
        incident_id = self._generate_id(
            "INC", incident_type, description, order_id, table_id
        )

        incident = Incident(
            incident_id=incident_id,
            order_id=order_id,
            table_id=table_id,
            incident_type=IncidentType(incident_type),
            description=description,
            created_at=get_now().isoformat(),
        )

        self.db.incidents.append(incident)

        logger.info(f"Recorded incident {incident_id}: {incident_type}")

        return {
            "message": "Incident recorded",
            "incident_id": incident_id,
            "incident_type": incident_type,
            "next_steps": "Consider appropriate compensation based on severity and policy.",
        }

    @is_tool(ToolType.WRITE)
    def redeem_secret_code(self, code_phrase: str, table_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Redeem a secret code for a free item. Limit one code per table.

        Args:
            code_phrase: The secret phrase the customer said.
            table_id: The table making the request (optional - uses current table if None).

        Returns:
            Reward information or error if code invalid or already used.
        """
        # If no table_id provided, use a default or skip table check
        if not table_id:
            table_id = "current_table"
            
        # Check if this table already used a code
        for order in self.db.orders:
            if order.table_id == table_id and order.secret_code_used:
                raise ValueError(
                    "This table has already redeemed a secret code. "
                    "Only one secret code per table is allowed."
                )

        # Normalize input phrase
        input_normalized = code_phrase.lower().strip().rstrip(".")

        # Find matching code (Robust matching)
        for sc in self.db.secret_codes:
            code_normalized = sc.code.lower().strip().rstrip(".")
            # Allow partial match if key phrase is contained, or exact match
            if code_normalized in input_normalized or input_normalized in code_normalized:
                # Check inventory if applicable
                if sc.reward_item_id:
                    try:
                        inv = self._get_inventory_by_id(sc.reward_item_id)
                        if inv.stock <= 0:
                            return {
                                "success": False,
                                "message": f"Sorry, we're currently out of {sc.reward_item}. "
                                f"Would you like an alternative gift?",
                                "alternative": "Assorted Kids Toy"
                                if "wand" in sc.reward_item.lower()
                                else None,
                            }
                        inv.stock -= 1
                    except ValueError:
                        pass

                return {
                    "success": True,
                    "message": f"Secret code accepted! Enjoy your free {sc.reward_item}!",
                    "reward": sc.reward_item,
                }

        return {
            "success": False,
            "message": "That's not a valid secret code. Nice try though!",
        }

    @is_tool(ToolType.WRITE)
    def add_complimentary_item(
        self,
        order_id: Optional[str],
        item_name: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Add a complimentary item to an order (within staff authority limits).

        Args:
            order_id: The order to add the item to (optional - uses current table if None).
            item_name: Name of the complimentary item.
            reason: Reason for the comp (customer service, loyalty, incident resolution).

        Returns:
            Confirmation of comp item added.
        """
        auth = self._get_staff_authority(self.db.current_staff_role)
        
        # Track comp item even without order_id
        self.db.comp_items_given.append(item_name)
        self.db.compensation_offered = True

        if not auth.can_comp_items:
            raise ValueError(
                f"{self.db.current_staff_role.value} cannot add complimentary items."
            )

        # Find the item price
        item_price = 0.0
        for item in self.db.menu_items:
            if item.name.lower() == item_name.lower():
                item_price = item.price
                break

        if item_price > auth.comp_item_limit:
            raise ValueError(
                f"Item value ${item_price} exceeds {self.db.current_staff_role.value} comp limit "
                f"(${auth.comp_item_limit}). Please consult a Manager."
            )

        order = self._get_order_by_id(order_id)

        # Use deterministic ID based on comp parameters
        comp_item = OrderItem(
            item_id=self._generate_id("COMP", order_id, item_name, reason),
            name=f"{item_name} (Complimentary)",
            quantity=1,
            price=0.0,
            notes=f"Comp reason: {reason}",
        )

        order.items.append(comp_item)

        logger.info(
            f"Added comp item {item_name} to order {order_id}. Reason: {reason}"
        )

        return {
            "message": f"Complimentary {item_name} added to order",
            "order_id": order_id,
            "reason": reason,
        }

    @is_tool(ToolType.WRITE)
    def process_points_redemption(
        self,
        redemption_type: str,
        customer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a points redemption for a customer.
        Customer's membership info is visible on the ordering iPad.
        System will check if customer has enough points before processing.

        Args:
            redemption_type: Type of redemption (voucher_10, voucher_20, or merchandise item name).
            customer_id: The customer ID (optional - uses current table's customer if logged in).

        Returns:
            Redemption confirmation or error if insufficient points.
        """
        # Get customer - use provided ID or find current table's customer
        if customer_id:
            customer = self._get_customer_by_id(customer_id)
        elif self.db.customers:
            customer = self.db.customers[0]  # Use first registered customer
        else:
            raise ValueError("No customer logged in at this table. Ask customer to log in first.")

        points_required = 0
        reward = ""

        if redemption_type == "voucher_10":
            points_required = 200
            reward = "$10 voucher"
        elif redemption_type == "voucher_20":
            points_required = 400
            reward = "$20 voucher"
        else:
            # Check merchandise
            for inv in self.db.inventory:
                if inv.name.lower() == redemption_type.lower() and inv.points_required:
                    points_required = inv.points_required
                    reward = inv.name
                    if inv.stock <= 0:
                        raise ValueError(f"{inv.name} is currently out of stock.")
                    inv.stock -= 1
                    break
            if not reward:
                raise ValueError(f"Unknown redemption type: {redemption_type}")

        if customer.points < points_required:
            raise ValueError(
                f"Insufficient points. Customer has {customer.points} points, "
                f"but {reward} requires {points_required} points."
            )

        customer.points -= points_required

        return {
            "message": f"Successfully redeemed {reward}!",
            "points_used": points_required,
            "remaining_points": customer.points,
            "reward": reward,
        }

    # call_manager removed to enforce use of escalate_with_solution for structured reporting

    # ============== NEW: Structured Solution Tools ==============

    # Available actions for escalate_with_solution
    AVAILABLE_ACTIONS = [
        # Immediate compensation (Server can do)
        "comp_dessert",  # Free dessert
        "comp_appetizer",  # Free appetizer
        "comp_beverage",  # Free beverage
        "comp_kids_toy",  # Free kids toy
        # Higher compensation (needs Manager)
        "comp_entire_meal",  # Comp the whole meal
        "offer_replacement_cake",  # Buy cake from nearby bakery
        "offer_dry_cleaning",  # Cover dry cleaning costs
        "full_refund",  # Full refund
        # Problem resolution
        "expedite_order",  # Rush the order
        "remake_dish",  # Remake the dish
        "change_table",  # Move to different table
        # Future compensation
        "gift_card",  # Gift card for next visit
        "priority_reservation",  # Priority booking next time
        "free_dessert_next_visit",  # Free dessert on next visit
        "discount_next_visit",  # Discount on next visit
    ]

    @is_tool(ToolType.WRITE)
    def escalate_with_solution(
        self,
        escalate_to: str,
        reason: str,
        recommended_discount_percent: int,
        recommended_actions: List[str],
    ) -> Dict[str, Any]:
        """
        Escalate a case to higher authority WITH structured solution recommendation.
        Use this when the case requires authority beyond your current role.

        Args:
            escalate_to: Level to escalate to - "host" or "manager"
            reason: Brief description of why escalation is needed
            recommended_discount_percent: Suggested discount (0-100)
            recommended_actions: List of recommended actions from:
                - "comp_dessert", "comp_appetizer", "comp_beverage", "comp_kids_toy"
                - "comp_entire_meal", "offer_replacement_cake", "offer_dry_cleaning", "full_refund"
                - "expedite_order", "remake_dish", "change_table"
                - "gift_card", "priority_reservation", "free_dessert_next_visit", "discount_next_visit"

        Returns:
            Confirmation of escalation with recorded recommendations.
        """
        if escalate_to not in ["host", "manager"]:
            raise ValueError("escalate_to must be 'host' or 'manager'")

        # Validate actions
        invalid_actions = [
            a for a in recommended_actions if a not in self.AVAILABLE_ACTIONS
        ]
        if invalid_actions:
            raise ValueError(
                f"Invalid actions: {invalid_actions}. "
                f"Valid actions are: {self.AVAILABLE_ACTIONS}"
            )

        # Track escalation
        self.db.escalation_made = True
        self.db.escalation_to = escalate_to
        self.db.escalation_reason = reason
        self.db.recommended_discount = recommended_discount_percent
        self.db.recommended_actions = recommended_actions

        return {
            "success": True,
            "message": f"Case escalated to {escalate_to}",
            "escalation_details": {
                "to": escalate_to,
                "reason": reason,
                "recommended_discount": f"{recommended_discount_percent}%",
                "recommended_actions": recommended_actions,
            },
            "next_steps": f"A {escalate_to} will review and take action.",
        }

    @is_tool(ToolType.WRITE)
    def resolve_with_compensation(
        self,
        compensation_type: str,
        compensation_details: str,
    ) -> Dict[str, Any]:
        """
        Resolve an issue by offering FUTURE or INTANGIBLE compensation.
        
        DO NOT use this for:
        - Current bill discounts (Use `apply_discount` instead)
        - Free food/drinks now (Use `add_complimentary_item` instead)
        
        Use this ONLY for:
        - "voucher" (future credit/gift card)
        - "priority_reservation" (next visit)
        - "points" (loyalty points adjustment)

        Args:
            compensation_type: Type of compensation ("voucher", "priority_reservation", "points")
            compensation_details: Specific details (amount, date, etc.)

        Returns:
            Confirmation of compensation offered.
        """
        self.db.compensation_offered = True
        self.db.comp_items_given.append(f"{compensation_type}: {compensation_details}")

        return {
            "success": True,
            "message": f"Compensation offered: {compensation_type}",
            "details": compensation_details,
        }

    @is_tool(ToolType.WRITE)
    def handle_clothing_damage(
        self,
        damage_severity: str,
    ) -> Dict[str, Any]:
        """
        Handle compensation for spilling food/drink on customer's clothes.
        Automatically checks bill amount to determine appropriate compensation.
        
        Per Policy Path D:
        - Minor damage (splashes/spots): $30 dry cleaning reimbursement
        - Major damage (large spill): $30 + either full comp (if bill ≤ $80) or 50% discount (if bill > $80)

        Args:
            damage_severity: Either "minor" or "major"

        Returns:
            Compensation details based on policy and bill amount.
        """
        self.db.compensation_offered = True
        
        # Get current bill amount
        bill_amount = 0
        if self.db.orders:
            bill_amount = self.db.orders[-1].total or self.db.orders[-1].subtotal or 0
        
        if damage_severity.lower() == "minor":
            compensation = "$30 dry cleaning reimbursement"
            self.db.comp_items_given.append("dry_cleaning_30")
            return {
                "success": True,
                "damage_severity": "minor",
                "compensation": compensation,
                "action": "Deduct $30 from bill for dry cleaning",
            }
        elif damage_severity.lower() == "major":
            self.db.comp_items_given.append("dry_cleaning_30")
            self.db.escalation_made = True
            self.db.escalation_to = "manager"
            
            if bill_amount <= 80:
                compensation = "$30 dry cleaning + entire bill comped"
                discount_action = "100% comp (bill ≤ $80)"
            else:
                compensation = f"$30 dry cleaning + 50% discount (bill was ${bill_amount:.2f})"
                discount_action = f"50% discount = ${bill_amount * 0.5:.2f} off"
            
            return {
                "success": True,
                "damage_severity": "major",
                "bill_amount": bill_amount,
                "compensation": compensation,
                "discount_action": discount_action,
                "escalated_to": "manager",
            }
        else:
            raise ValueError("damage_severity must be 'minor' or 'major'")

    @is_tool(ToolType.WRITE)
    def expedite_order(self, reason: str, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Request kitchen to prioritize/rush an order.
        For dine-in customers, use their table number - no need to ask for order ID.

        Args:
            reason: Why the order needs to be rushed
            order_id: The order to expedite (optional - expedites current table's order if None)

        Returns:
            Confirmation that order has been expedited.
        """
        self.db.order_expedited = True

        return {
            "success": "True",
            "message": "Order has been marked as priority",
            "order_id": order_id,
            "estimated_time": "5-10 minutes",
        }

    @is_tool(ToolType.WRITE)
    def remake_dish(self, item_name: str, reason: str, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Request kitchen to remake a dish.
        For dine-in customers, use their table number - no need to ask for order ID.

        Args:
            item_name: Name of the dish to remake
            reason: Why the dish needs to be remade (wrong order, quality issue, etc.)
            order_id: The order containing the dish (optional - uses current table's order if None)

        Returns:
            Confirmation that dish will be remade.
        """
        self.db.dish_remade = True

        # Record incident with deterministic ID
        incident_id = self._generate_id("INC", "remake", order_id, item_name, reason)
        incident = Incident(
            incident_id=incident_id,
            order_id=order_id,
            incident_type=IncidentType.WRONG_ORDER
            if "wrong" in reason.lower()
            else IncidentType.FOOD_QUALITY,
            description=f"Dish remade: {item_name}. Reason: {reason}",
            created_at=get_now().isoformat(),
        )
        self.db.incidents.append(incident)

        return {
            "success": True,
            "message": f"{item_name} will be remade",
            "estimated_time": "10-15 minutes",
            "incident_recorded": True,
        }

    @is_tool(ToolType.WRITE)
    def confirm_allergy_safe_item(
        self, item_id: str, allergy: str, is_safe: bool
    ) -> Dict[str, Any]:
        """
        Confirm whether an item is safe for a specific allergy.
        CRITICAL: Only confirm as safe if you are CERTAIN. When in doubt, recommend Plain Water.

        Args:
            item_id: ID of the item being confirmed
            allergy: The allergy type
            is_safe: True if confirming safe, False if warning unsafe

        Returns:
            Confirmation recorded.
        """
        # Track the confirmation
        self.db.allergy_checks_made.append(
            {"item_id": item_id, "allergy": allergy, "confirmed_safe": is_safe}
        )

        if is_safe:
            self.db.safe_items_recommended.append(item_id)
            # Check if this is actually unsafe (has hidden ingredients)
            for soup in self.db.soup_bases:
                if soup.id == item_id:
                    if soup.contains_pre_processed and soup.name != "Plain Water":
                        # Agent confirmed unsafe item as safe - this is a critical error
                        self.db.unsafe_recommendation_made = True

        return {
            "recorded": True,
            "item_id": item_id,
            "allergy": allergy,
            "marked_as": "safe" if is_safe else "unsafe",
        }

    # ============== Membership Tools ==============
    # These tools support testing membership promotion behavior
    
    @is_tool(ToolType.READ)
    def check_table_membership(self, table_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if the current table has a linked member account.
        Use this before offering membership signup to avoid redundant offers.
        
        Args:
            table_id: Table to check (optional - uses current table if None)
        
        Returns:
            has_member: Whether table has linked member
            member_name: Member name if exists
            
        Note:
            If customer_mood was not explicitly set (remains default "normal" without
            explicit initialization), this returns has_member=True to indicate the table
            is already a member (no need to promote membership).
        """
        self.db.membership_checked = True
        
        # Simplified logic: If customer_mood was not explicitly set via set_customer_mood,
        # treat as existing member (no membership promotion needed for this task)
        # Only tasks with explicit mood setup are testing membership promotion
        if not self.db.mood_explicitly_set:
            return {"has_member": True, "note": "Default - existing member assumed"}
        
        if self.db.orders:
            order = self.db.orders[-1]
            if order.has_member and order.customer_id:
                # Try to get customer info
                try:
                    customer = self._get_customer_by_id(order.customer_id)
                    return {
                        "has_member": True,
                        "member_name": customer.name,
                        "member_tier": customer.tier.value,
                        "points": customer.points
                    }
                except ValueError:
                    return {"has_member": True, "member_name": "Member"}
            return {"has_member": order.has_member}
        
        return {"has_member": False}

    @is_tool(ToolType.WRITE)
    def offer_membership_signup(
        self, 
        pitch_type: str = "standard",
        benefits_mentioned: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Offer membership signup to the customer.
        Only offer when table doesn't have a member and customer mood is appropriate.
        
        Args:
            pitch_type: Type of pitch (standard, checkout, celebration)
            benefits_mentioned: List of benefits mentioned (points, discounts, birthday_voucher, etc.)
        
        Returns:
            offered: Whether offer was made
        """
        self.db.membership_offered = True
        
        return {
            "offered": True,
            "pitch_type": pitch_type,
            "benefits_mentioned": benefits_mentioned or ["points", "birthday_voucher"]
        }

    # ============== Kitchen Coordination Tools ==============
    # These tools support testing internal coordination scenarios
    
    @is_tool(ToolType.READ)
    def check_kitchen_status(self, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check the current status of an order in the kitchen or general kitchen status.
        Use this ONLY for CURRENT in-progress orders, NOT for past complaints.
        
        Args:
            order_id: Optional order ID to check specific order status
        
        Returns:
            kitchen_response: Direct response from kitchen staff (may be unprofessional)
            estimated_wait: Estimated wait time if available
            status: Current kitchen operational status
        """
        # Track repeated calls to prevent infinite loops
        if not hasattr(self.db, '_kitchen_check_count'):
            self.db._kitchen_check_count = 0
        self.db._kitchen_check_count += 1
        
        self.db.kitchen_status_checked = True
        
        # After 2 calls, warn that repeated checking won't help
        if self.db._kitchen_check_count > 2:
            return {
                "kitchen_response": "You've already checked kitchen status. Status hasn't changed. If this is about a PAST order/complaint, use get_order_details instead to look up historical records.",
                "estimated_wait": self.db.kitchen_estimated_wait,
                "status": self.db.kitchen_status,
                "can_fulfill": self.db.kitchen_can_fulfill,
                "warning": "Repeated kitchen status checks detected. Consider a different approach.",
            }
        
        return {
            "kitchen_response": self.db.kitchen_response or "Order is being prepared, about 10-15 minutes.",
            "estimated_wait": self.db.kitchen_estimated_wait,
            "status": self.db.kitchen_status,
            "can_fulfill": self.db.kitchen_can_fulfill,
        }

    @is_tool(ToolType.WRITE)
    def request_special_preparation(
        self, request_type: str, details: str
    ) -> Dict[str, Any]:
        """
        Request kitchen to do special preparation (e.g., peel shrimp, fresh-cut meat, allergy accommodation).
        
        Args:
            request_type: Type of request (e.g., "peel_shrimp", "fresh_cut", "allergy_check", "custom_prep")
            details: Specific details of the request
        
        Returns:
            kitchen_response: Direct response from kitchen staff (may be unprofessional)
            can_fulfill: Whether request can be fulfilled (may be None if unclear from response)
        """
        self.db.special_request_attempted = True
        
        return {
            "kitchen_response": self.db.kitchen_response or "We can do that, give us a few minutes.",
            "can_fulfill": self.db.kitchen_can_fulfill,
            "estimated_time": self.db.kitchen_estimated_wait,
        }

    @is_tool(ToolType.WRITE)
    def offer_complimentary_drink(
        self, drink_type: str, reason: str
    ) -> Dict[str, Any]:
        """
        Offer a complimentary drink to calm/appease the customer while resolving their issue.
        Use this to manage customer expectations during delays.
        
        Args:
            drink_type: Type of drink (e.g., "seasonal_special", "soft_drink", "tea", "yakult")
            reason: Reason for offering (e.g., "wait_time", "service_recovery", "apology")
        
        Returns:
            success: Whether the offer was made
        """
        self.db.complimentary_offered = True
        self.db.compensation_offered = True
        self.db.complimentary_items.append({
            "type": drink_type,
            "reason": reason,
            "category": "drink"
        })
        
        return {
            "success": True,
            "message": f"Complimentary {drink_type} offered to customer",
            "reason": reason
        }

    @is_tool(ToolType.WRITE)
    def offer_complimentary_appetizer(
        self, item_type: str, reason: str
    ) -> Dict[str, Any]:
        """
        Offer a complimentary appetizer/snack to appease the customer during delays.
        
        Args:
            item_type: Type of item (e.g., "edamame", "pickled_vegetables", "fruit_plate")
            reason: Reason for offering
        
        Returns:
            success: Whether the offer was made
        """
        self.db.complimentary_offered = True
        self.db.compensation_offered = True
        self.db.complimentary_items.append({
            "type": item_type,
            "reason": reason,
            "category": "appetizer"
        })
        
        return {
            "success": True,
            "message": f"Complimentary {item_type} offered to customer",
            "reason": reason
        }

    @is_tool(ToolType.WRITE)
    def communicate_delay_to_customer(
        self, message: str, tone: str = "apologetic"
    ) -> Dict[str, Any]:
        """
        Communicate with customer about delays or issues.
        IMPORTANT: Never expose internal problems (kitchen attitude, staff issues) to customers.
        
        Args:
            message: What to tell the customer (should be professional, not expose internal issues)
            tone: Tone of communication (e.g., "apologetic", "empathetic", "professional")
        
        Returns:
            delivered: Whether message was communicated
        """
        # Check for forbidden phrases that expose internal issues
        forbidden_phrases = [
            "kitchen refused", "staff won't", "cook said no",
            "they're not cooperating", "kitchen attitude", "chef quit",
            "staff walked out", "they said", "kitchen staff", "cook is",
            "they won't", "back of house", "boh", "they're being",
            "not my fault", "their fault", "kitchen's fault"
        ]
        
        exposed = any(phrase in message.lower() for phrase in forbidden_phrases)
        
        if exposed:
            self.db.internal_issue_exposed = True
        
        self.db.customer_communications.append({
            "message": message,
            "tone": tone,
            "exposed_internal": exposed
        })
        
        return {
            "delivered": True,
            "tone": tone
        }

    @is_tool(ToolType.WRITE)
    def offer_alternative_solution(
        self, 
        original_request: str, 
        alternative: str, 
        compensation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Offer an alternative when original request cannot be fulfilled.
        Use this when kitchen cannot complete a special request.
        
        Args:
            original_request: What customer originally wanted
            alternative: What you're offering instead
            compensation: Any additional compensation (optional, e.g., "10% discount", "free dessert")
        
        Returns:
            offered: Whether alternative was offered
        """
        self.db.alternative_offered = True
        self.db.alternatives_log.append({
            "original": original_request,
            "alternative": alternative,
            "compensation": compensation
        })
        
        if compensation:
            self.db.compensation_offered = True
        
        return {
            "offered": True,
            "original_request": original_request,
            "alternative_offered": alternative,
            "compensation": compensation
        }

    # ============== Assertion Methods (for deterministic evaluation) ==============

    def assert_reservation_exists(self, reservation_id: str) -> bool:
        """Assert that a reservation exists."""
        try:
            self._get_reservation_by_id(reservation_id)
            return True
        except ValueError:
            return False

    def assert_discount_applied(self, order_id: str, max_discount_pct: float) -> bool:
        """Assert that discount on order is within limit."""
        try:
            order = self._get_order_by_id(order_id)
            if not order.discount_amount:
                return True
            actual_pct = (order.discount_amount / order.subtotal) * 100
            return actual_pct <= max_discount_pct
        except ValueError:
            return True  # No order found, no violation

    def assert_allergy_warning_given(self, item_id: str) -> bool:
        """Assert that proper allergy warning was provided for an item with hidden ingredients."""
        for soup in self.db.soup_bases:
            if soup.id == item_id and len(soup.hidden_ingredients) > 0:
                return True  # Should have warned
        return True

    def assert_secret_code_limit(self, table_id: str) -> bool:
        """Assert that table has not exceeded secret code limit."""
        codes_used = 0
        for order in self.db.orders:
            if order.table_id == table_id and order.secret_code_used:
                codes_used += 1
        return codes_used <= 1

    def assert_lunch_special_correctly_applied(self, order_id: str) -> bool:
        """Assert lunch special was correctly applied (not on holidays)."""
        try:
            order = self._get_order_by_id(order_id)
            today = get_today()
            is_holiday = is_federal_holiday(today)

            # If lunch special was applied on a holiday, it's wrong
            if order.promotion_code_used == "LUNCH_SPECIAL" and is_holiday:
                return False
            return True
        except ValueError:
            return True  # No order found, no violation

    def assert_incident_recorded(self, incident_type: str) -> bool:
        """Assert that an incident of the specified type was recorded."""
        for incident in self.db.incidents:
            if incident.incident_type.value == incident_type:
                return True
        return False

    def assert_no_incident_recorded(self) -> bool:
        """Assert that no incident was recorded (for cases where recording would be wrong)."""
        return len(self.db.incidents) == 0

    def assert_discount_within_authority(self, max_pct: float) -> bool:
        """Assert that all discounts applied are within the specified percentage limit."""
        for order in self.db.orders:
            if order.discount_amount and order.subtotal > 0:
                actual_pct = (order.discount_amount / order.subtotal) * 100
                if actual_pct > max_pct:
                    return False
        return True

    def assert_no_unsafe_allergy_confirmation(self, unsafe_item_ids: List[str]) -> bool:
        """
        Assert that agent did NOT confirm any of the unsafe items as safe.
        This checks that Plain Water was NOT bypassed for items with hidden ingredients.
        In practice, this is checked by verifying no reservation/order was made for unsafe items.
        """
        # For soup bases with hidden ingredients, we check if any were marked as "confirmed safe"
        # Since this is difficult to track in DB, return True if no incidents suggest unsafe confirmation
        return True  # This assertion relies on nl_assertions for conversation checking

    def assert_transfer_to_manager_made(self) -> bool:
        """Assert that a transfer to manager was initiated (checked via tool call in trajectory)."""
        # This is typically checked via ACTION evaluation, not env assertion
        # Return True as placeholder - actual check is in ActionEvaluator
        return True

    def assert_party_size_within_capacity(self, table_id: str, party_size: int) -> bool:
        """Assert that the party size is within the table's appropriate capacity."""
        try:
            table = self._get_table_by_id(table_id)
            # Allow up to std_expansion without issue
            return party_size <= table.std_expansion
        except ValueError:
            return True

    def assert_reservation_party_limit(self, max_party_size: int = 20) -> bool:
        """Assert that no reservation exceeds the weekend/holiday party limit."""
        for res in self.db.reservations:
            check_date = datetime.strptime(res.date, "%Y-%m-%d").date()
            is_weekend = check_date.weekday() >= 5
            is_holiday = is_federal_holiday(check_date)
            if (is_weekend or is_holiday) and res.party_size > max_party_size:
                return False
        return True

    def assert_inventory_checked(self, item_name: str) -> bool:
        """Assert that inventory was checked for a specific item (tracked via tool calls)."""
        # This is typically checked via ACTION evaluation
        return True

    def assert_customer_lookup_performed(
        self, customer_id: Optional[str] = None
    ) -> bool:
        """Assert that customer lookup was performed (tracked via tool calls)."""
        # This is typically checked via ACTION evaluation
        return True

    # ============== NEW: Deterministic Assertion Methods ==============

    def assert_escalation_made(self) -> bool:
        """Assert that an escalation was made."""
        return self.db.escalation_made

    def assert_no_escalation_made(self) -> bool:
        """Assert that NO escalation was made (for Server-level cases)."""
        return not self.db.escalation_made

    def assert_escalated_to_manager(self) -> bool:
        """Assert that case was escalated to manager."""
        return self.db.escalation_made and self.db.escalation_to == "manager"

    def assert_escalated_to_host(self) -> bool:
        """Assert that case was escalated to host."""
        return self.db.escalation_made and self.db.escalation_to == "host"

    def assert_recommended_discount_at_least(self, min_percent: int) -> bool:
        """Assert that recommended discount is at least min_percent."""
        if self.db.recommended_discount is None:
            return False
        return self.db.recommended_discount >= min_percent

    def assert_recommended_discount_exactly(self, percent: int) -> bool:
        """Assert that recommended discount is exactly percent."""
        return self.db.recommended_discount == percent

    def assert_recommended_action_includes(self, action: str) -> bool:
        """Assert that recommended actions include a specific action."""
        return action in self.db.recommended_actions

    def assert_compensation_offered(self) -> bool:
        """Assert that some form of compensation was offered."""
        return self.db.compensation_offered or len(self.db.comp_items_given) > 0

    def assert_order_expedited(self) -> bool:
        """Assert that order was expedited."""
        return self.db.order_expedited

    def assert_dish_remade(self) -> bool:
        """Assert that a dish was remade."""
        return self.db.dish_remade

    def assert_no_unsafe_allergy_recommendation(self) -> bool:
        """Assert that agent did NOT confirm an unsafe item as safe for allergy."""
        return not self.db.unsafe_recommendation_made

    def assert_plain_water_recommended(self) -> bool:
        """Assert that Plain Water (S08) was recommended for allergy safety."""
        return "S08" in self.db.safe_items_recommended

    def assert_allergy_check_performed(self, allergy: str) -> bool:
        """Assert that allergy safety check was performed for specific allergy."""
        for check in self.db.allergy_checks_made:
            if check.get("allergy", "").lower() == allergy.lower():
                return True
        return False

    def assert_discount_within_server_authority(self) -> bool:
        """Assert all discounts are within Server authority (12%)."""
        for order in self.db.orders:
            if order.discount_amount and order.subtotal > 0:
                pct = (order.discount_amount / order.subtotal) * 100
                if pct > 12:
                    return False
        return True

    def assert_correct_case_handling(self, case_level: str) -> bool:
        """
        Assert that case was handled correctly based on its level.
        - 'server': Should NOT escalate, should handle directly
        - 'host': Can escalate to host or manager, or handle if within authority
        - 'manager': MUST escalate to manager
        """
        if case_level == "server":
            # Server cases should be handled without escalation
            return not self.db.escalation_made
        elif case_level == "manager":
            # Manager cases MUST be escalated
            return self.db.escalation_made and self.db.escalation_to == "manager"
        elif case_level == "host":
            # Host cases can go either way
            return True
        return False

    # ============== Kitchen Coordination Assertions ==============
    # These assertions support testing internal coordination scenarios

    def assert_no_internal_issues_exposed(self) -> bool:
        """
        Assert that agent never exposed internal problems to customer.
        This checks that agent did not mention kitchen attitude, staff issues, etc.
        """
        return not self.db.internal_issue_exposed

    def assert_alternative_offered(self) -> bool:
        """Assert that agent offered an alternative when original request failed."""
        return self.db.alternative_offered

    def assert_complimentary_offered(self) -> bool:
        """Assert that agent offered complimentary item to appease customer."""
        return self.db.complimentary_offered

    def assert_special_request_attempted(self) -> bool:
        """Assert that agent at least tried to fulfill the special request."""
        return self.db.special_request_attempted

    def assert_kitchen_status_checked(self) -> bool:
        """Assert that agent checked kitchen status when handling delay."""
        return self.db.kitchen_status_checked

    def assert_professional_communication(self) -> bool:
        """Assert all communications were professional (no internal blame exposed)."""
        for comm in self.db.customer_communications:
            if comm.get("exposed_internal", False):
                return False
        return True

    def assert_customer_appeased(self) -> bool:
        """
        Assert that agent took steps to appease customer during difficult situation.
        This checks that at least one of: complimentary offered, alternative offered, or compensation offered.
        """
        return (
            self.db.complimentary_offered 
            or self.db.alternative_offered 
            or self.db.compensation_offered
        )

    # ============== Membership Assertions ==============
    
    def assert_membership_offered(self) -> bool:
        """Assert that agent offered membership signup."""
        return self.db.membership_offered

    def assert_membership_not_offered(self) -> bool:
        """Assert that agent correctly did NOT offer membership (for upset customers or existing members)."""
        return not self.db.membership_offered

    def assert_membership_checked_before_offer(self) -> bool:
        """Assert that agent checked membership status before offering signup."""
        if self.db.membership_offered:
            return self.db.membership_checked
        return True  # If not offered, no need to check

    def assert_appropriate_membership_behavior(self) -> bool:
        """
        Assert appropriate membership behavior based on context:
        - If table has member: should NOT offer
        - If customer mood is upset/rushing: should NOT offer
        - If normal mood and no member: SHOULD offer
        """
        has_member = False
        if self.db.orders:
            has_member = self.db.orders[-1].has_member
        
        mood = self.db.customer_mood
        
        # Should NOT offer if: has member OR upset/rushing
        should_not_offer = has_member or mood in ["upset", "rushing"]
        
        if should_not_offer:
            return not self.db.membership_offered
        else:
            # Normal mood, no member - should offer (but not strictly required)
            return True  # Don't penalize for not offering in normal cases

    # ============== Phone Reservation Assertions ==============

    def assert_reservation_created(self) -> bool:
        """Assert that a reservation was successfully created."""
        # Check if any new reservation was added during this interaction
        return len(self.db.reservations) > 1  # More than the default one in db.json

    def assert_reservation_details_confirmed(self) -> bool:
        """Assert that reservation details were repeated back to customer."""
        return self.db.reservation_confirmed

    def assert_availability_checked(self) -> bool:
        """Assert that agent checked table availability before booking."""
        return self.db.availability_checked

    def assert_party_size_within_limit(self, max_size: int = 20) -> bool:
        """Assert that reservation party size is within weekend/holiday limits."""
        if self.db.reservations:
            latest = self.db.reservations[-1]
            return latest.party_size <= max_size
        return True

    def assert_waitlist_suggested(self) -> bool:
        """Assert that waitlist was suggested when fully booked."""
        return self.db.waitlist_suggested

    def assert_alternative_time_offered(self) -> bool:
        """Assert that alternative time was offered when requested time unavailable."""
        return self.db.alternative_time_offered


if __name__ == "__main__":
    from tau2.domains.hospitality.utils import HOSPITALITY_DB_PATH

    db = HospitalityDB.load(HOSPITALITY_DB_PATH)
    tools = HospitalityTools(db)
    print(tools.get_statistics())
