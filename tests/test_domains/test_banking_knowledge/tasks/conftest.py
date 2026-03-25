"""Shared fixtures for task tests in the knowledge domain."""

from tau2.domains.banking_knowledge.data_model import DatabaseTable, TransactionalDB
from tau2.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
from tau2.environment.environment import Environment


def create_amara_db(auto_resolve: bool = True) -> TransactionalDB:
    """Create a TransactionalDB with Amara Okonkwo's data for task_026/027."""
    return TransactionalDB(
        users=DatabaseTable(
            data={
                "755bcb4d5d": {
                    "name": "Amara Okonkwo",
                    "user_id": "755bcb4d5d",
                    "address": "305 Magnolia Street, Houston, TX 77002",
                    "email": "dancing_star_amara@icloud.com",
                    "phone_number": "713-555-0963",
                    "date_of_birth": "08/11/1997",
                },
            }
        ),
        credit_card_transaction_history=DatabaseTable(
            data={
                "txn_a8f1c2d3e403": {
                    "transaction_id": "txn_a8f1c2d3e403",
                    "user_id": "755bcb4d5d",
                    "credit_card_type": "Business Silver Rewards Card",
                    "merchant_name": "JetBlue Airways",
                    "transaction_amount": "$315.00",
                    "transaction_date": "03/22/2025",
                    "category": "Travel",
                    "status": "COMPLETED",
                    "rewards_earned": "3150 points",  # Incorrect - should be 6300 with promo
                },
                "txn_b7e2d4c5f506": {
                    "transaction_id": "txn_b7e2d4c5f506",
                    "user_id": "755bcb4d5d",
                    "credit_card_type": "Silver Rewards Card",
                    "merchant_name": "GitHub Enterprise",
                    "transaction_amount": "$255.00",
                    "transaction_date": "05/18/2025",
                    "category": "Software",
                    "status": "COMPLETED",
                    "rewards_earned": "2550 points",  # Incorrect - should be 1020 (4%)
                },
                "txn_a8f1c2d3e410": {
                    "transaction_id": "txn_a8f1c2d3e410",
                    "user_id": "755bcb4d5d",
                    "credit_card_type": "Business Silver Rewards Card",
                    "merchant_name": "Southwest Airlines",
                    "transaction_amount": "$380.00",
                    "transaction_date": "08/25/2025",
                    "category": "Travel",
                    "status": "COMPLETED",
                    "rewards_earned": "1520 points",  # Incorrect - should be 3800
                },
                "txn_a8f1c2d3e411": {
                    "transaction_id": "txn_a8f1c2d3e411",
                    "user_id": "755bcb4d5d",
                    "credit_card_type": "Business Silver Rewards Card",
                    "merchant_name": "Zoom Video",
                    "transaction_amount": "$149.99",
                    "transaction_date": "09/15/2025",
                    "category": "Software",
                    "status": "COMPLETED",
                    "rewards_earned": "600 points",  # Incorrect - should be 1500
                },
            }
        ),
        credit_card_accounts=DatabaseTable(
            data={
                "cc_755bcb4d5d_bsilver": {
                    "account_id": "cc_755bcb4d5d_bsilver",
                    "user_id": "755bcb4d5d",
                    "card_type": "Business Silver Rewards Card",
                    "date_of_account_open": "02/13/2025",
                    "current_balance": "$3,105.12",
                    "reward_points": "37848 points",
                },
                "cc_755bcb4d5d_silver": {
                    "account_id": "cc_755bcb4d5d_silver",
                    "user_id": "755bcb4d5d",
                    "card_type": "Silver Rewards Card",
                    "date_of_account_open": "01/20/2025",
                    "current_balance": "$2,465.79",
                    "reward_points": "9875 points",
                },
            }
        ),
        cash_back_disputes=DatabaseTable(data={}),
        user_discoverable_tools=DatabaseTable(data={}),
        user_discoverable_tool_calls=DatabaseTable(data={}),
        agent_discoverable_tools=DatabaseTable(data={}),
        verification_history=DatabaseTable(data={}),
        task_config=DatabaseTable(
            data={"dispute_settings": {"auto_resolve_disputes": auto_resolve}}
        ),
    )


def create_fatima_db(auto_resolve: bool = True) -> TransactionalDB:
    """Create a TransactionalDB with Fatima Al-Hassan's data for task_028/029."""
    return TransactionalDB(
        users=DatabaseTable(
            data={
                "890389b165": {
                    "name": "Fatima Al-Hassan",
                    "user_id": "890389b165",
                    "address": "1923 Oak Park Boulevard, Detroit, MI 48226",
                    "email": "coffeelover_fati@protonmail.com",
                    "phone_number": "313-555-0246",
                    "date_of_birth": "12/05/1993",
                },
            }
        ),
        credit_card_transaction_history=DatabaseTable(
            data={
                "txn_57ecc6da56c2": {
                    "transaction_id": "txn_57ecc6da56c2",
                    "user_id": "890389b165",
                    "credit_card_type": "Crypto-Cash Back",
                    "merchant_name": "Trader Joe's",
                    "transaction_amount": "$47.83",
                    "transaction_date": "11/01/2025",
                    "category": "Groceries",
                    "status": "COMPLETED",
                    "rewards_earned": "47 points",  # Incorrect - should be 95 (2%)
                },
                "txn_d80aef98f532": {
                    "transaction_id": "txn_d80aef98f532",
                    "user_id": "890389b165",
                    "credit_card_type": "Business Platinum Rewards Card",
                    "merchant_name": "United Airlines",
                    "transaction_amount": "$347.62",
                    "transaction_date": "11/02/2025",
                    "category": "Travel",
                    "status": "COMPLETED",
                    "rewards_earned": "521 points",  # Incorrect - should be 1390 (4%)
                },
                "txn_896ac64b98d7": {
                    "transaction_id": "txn_896ac64b98d7",
                    "user_id": "890389b165",
                    "credit_card_type": "EcoCard",
                    "merchant_name": "Patagonia",
                    "transaction_amount": "$128.47",
                    "transaction_date": "11/04/2025",
                    "category": "Green",
                    "status": "COMPLETED",
                    "rewards_earned": "128 points",  # Incorrect - should be 642 (5%)
                },
                "txn_adea68821a1d": {
                    "transaction_id": "txn_adea68821a1d",
                    "user_id": "890389b165",
                    "credit_card_type": "Silver Rewards Card",
                    "merchant_name": "Marriott Hotels",
                    "transaction_amount": "$167.34",
                    "transaction_date": "11/07/2025",
                    "category": "Travel",
                    "status": "COMPLETED",
                    "rewards_earned": "167 points",  # Incorrect - should be 669 (4%)
                },
                "txn_0be1ccc37761": {
                    "transaction_id": "txn_0be1ccc37761",
                    "user_id": "890389b165",
                    "credit_card_type": "Business Platinum Rewards Card",
                    "merchant_name": "LinkedIn Ads",
                    "transaction_amount": "$512.47",
                    "transaction_date": "11/23/2025",
                    "category": "Media",
                    "status": "COMPLETED",
                    "rewards_earned": "768 points",  # Incorrect - should be 2049 (4%)
                },
                "txn_e647e242ce96": {
                    "transaction_id": "txn_e647e242ce96",
                    "user_id": "890389b165",
                    "credit_card_type": "Business Platinum Rewards Card",
                    "merchant_name": "Google Ads",
                    "transaction_amount": "$187.56",
                    "transaction_date": "11/18/2025",
                    "category": "Media",
                    "status": "COMPLETED",
                    "rewards_earned": "1875 points",  # Incorrect - should be 750 (4%)
                },
            }
        ),
        credit_card_accounts=DatabaseTable(
            data={
                "cc_890389b165_crypto": {
                    "account_id": "cc_890389b165_crypto",
                    "user_id": "890389b165",
                    "card_type": "Crypto-Cash Back",
                    "date_of_account_open": "06/10/2024",
                    "current_balance": "$1,389.98",
                    "reward_points": "2631 points",
                },
                "cc_890389b165_bplat": {
                    "account_id": "cc_890389b165_bplat",
                    "user_id": "890389b165",
                    "card_type": "Business Platinum Rewards Card",
                    "date_of_account_open": "02/28/2024",
                    "current_balance": "$4,212.70",
                    "reward_points": "14223 points",
                },
                "cc_890389b165_eco": {
                    "account_id": "cc_890389b165_eco",
                    "user_id": "890389b165",
                    "card_type": "EcoCard",
                    "date_of_account_open": "07/01/2024",
                    "current_balance": "$927.36",
                    "reward_points": "1994 points",
                },
                "cc_890389b165_silver": {
                    "account_id": "cc_890389b165_silver",
                    "user_id": "890389b165",
                    "card_type": "Silver Rewards Card",
                    "date_of_account_open": "04/15/2024",
                    "current_balance": "$1,173.27",
                    "reward_points": "4928 points",
                },
            }
        ),
        cash_back_disputes=DatabaseTable(data={}),
        user_discoverable_tools=DatabaseTable(data={}),
        user_discoverable_tool_calls=DatabaseTable(data={}),
        agent_discoverable_tools=DatabaseTable(data={}),
        verification_history=DatabaseTable(data={}),
        task_config=DatabaseTable(
            data={"dispute_settings": {"auto_resolve_disputes": auto_resolve}}
        ),
    )


def create_environment(db: TransactionalDB) -> Environment:
    """Create a test environment with the given database.

    Uses the base KnowledgeTools toolkit (no retrieval tools) with an empty
    policy. This is sufficient for tests that exercise agent/user tools
    and do not require retrieval capabilities.
    """
    tools = KnowledgeTools(db)
    user_tools = KnowledgeUserTools(db)
    return Environment(
        domain_name="banking_knowledge",
        policy="",
        tools=tools,
        user_tools=user_tools,
    )
