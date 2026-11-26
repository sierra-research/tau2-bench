"""CRM domain tools for interacting with Twenty CRM via MCP."""

import os
import json
import time
import requests
from loguru import logger
from typing import Optional
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.crm.data_model import CRMDB, Person, Company, Opportunity, Task


_SELECT_FIELDS = {
    "companies": ["id", "name"],
    "people": ["id", "name", "companyId"],
    "opportunities": ["id", "name", "pointOfContactId", "companyId", "stage"],
    "tasks": ["id", "title", "status"],
}


class CRMMCPTools(ToolKitBase):
    """CRM tools for relationship management based on Commitment-Trust Theory.
    
    This class connects to Twenty CRM's MCP server and provides tools for
    managing people, companies, opportunities, and tasks while tracking
    trust and commitment metrics.
    
    Credentials are loaded from .env file in the tau2-bench-copy directory.
    """
    
    db: CRMDB
    mcp_url: str
    mcp_token: str
    
    def __init__(self, db: CRMDB) -> None:
        super().__init__(db)
        
        # Load credentials from tau2-bench-copy/.env file
        self.mcp_url, self.mcp_token = self._load_mcp_credentials()

        # Twenty limits find_many_* to 100 calls per 60s. Pace and retry so a
        # rate-limited read is not scored as a model tool error.
        self._last_mcp_call = 0.0
        self._mcp_min_interval = float(
            os.environ.get("TWENTY_MIN_INTERVAL", 0.7)
        )
        self._mcp_max_retries = int(
            os.environ.get("TWENTY_MAX_RETRIES", 5)
        )
        self._mcp_retry_wait = int(os.environ.get("TWENTY_RETRY_WAIT", 60))

        if not self.mcp_token or not self.mcp_url:
            raise ValueError(
                "TWENTY_MCP_TOKEN/TWENTY_MCP_URL not found. Please create .env file "
                "in tau2-bench-copy directory with TWENTY_MCP_TOKEN and TWENTY_MCP_URL"
            )
    
    def _load_mcp_credentials(self) -> tuple[str, str]:
        """Load MCP credentials from .env file in tau2-bench-copy directory.
        
        Returns:
            Tuple of (mcp_url, mcp_token)
        """
        # Navigate to tau2-bench-copy directory (parent of src)
        # Look for .env file in the project root
        crm_dir = os.path.dirname(os.path.abspath(__file__))
        tau2_bench_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(crm_dir)))
        )
        env_file = os.path.join(tau2_bench_dir, ".env")

        mcp_url = None
        mcp_token = None
        
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        if key == "TWENTY_MCP_URL":
                            mcp_url = value
                        elif key == "TWENTY_MCP_TOKEN":
                            mcp_token = value
        
        return mcp_url, mcp_token
    
    def _call_mcp(self, method_name: str, arguments: dict) -> dict:
        """Call Twenty CRM MCP server.
        
        Args:
            method_name: The MCP tool name (e.g., "create_one_person")
            arguments: Arguments to pass to the tool
            
        Returns:
            Parsed JSON response from MCP server
        """
        arguments = dict(arguments or {})
        if method_name.startswith("find_many_") and "select" not in arguments:
            object_name = method_name[len("find_many_") :]
            arguments["select"] = _SELECT_FIELDS.get(object_name, ["id"])

        headers = {
            "Authorization": f"Bearer {self.mcp_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "execute_tool",
                "arguments": {
                    "toolName": method_name,
                    "arguments": arguments
                }
            },
            "id": 1
        }
        
        parsed: object = {}
        for attempt in range(1, self._mcp_max_retries + 1):
            wait = self._mcp_min_interval - (
                time.monotonic() - self._last_mcp_call
            )
            if wait > 0:
                time.sleep(wait)

            response = requests.post(
                self.mcp_url, json=payload, headers=headers, timeout=120
            )
            self._last_mcp_call = time.monotonic()

            retry_reason = None
            if response.status_code == 429:
                retry_reason = "HTTP 429"
            else:
                response.raise_for_status()

                result = response.json()
                parsed = result
                if "result" in result and "content" in result["result"]:
                    text_content = result["result"]["content"][0]["text"]
                    try:
                        parsed = json.loads(text_content)
                    except json.JSONDecodeError:
                        parsed = {"raw": text_content}

                if (
                    isinstance(parsed, dict)
                    and parsed.get("success") is False
                ):
                    message = str(parsed.get("message") or "").lower()
                    if "limit reached" in message:
                        retry_reason = "rate limit"
                    elif method_name.startswith("find_many_"):
                        retry_reason = "find failure"

            if retry_reason and attempt < self._mcp_max_retries:
                logger.warning(
                    f"MCP {retry_reason} on {method_name}; sleeping "
                    f"{self._mcp_retry_wait}s "
                    f"(attempt {attempt}/{self._mcp_max_retries})"
                )
                time.sleep(self._mcp_retry_wait)
                continue
            return parsed

        return parsed

    @staticmethod
    def _map_stage(stage: str) -> str:
        """Map a domain relationship stage to a Twenty opportunity stage."""
        mapping = {
            "prospecting": "NEW",
            "new": "NEW",
            "screening": "SCREENING",
            "qualified": "SCREENING",
            "meeting": "MEETING",
            "proposal": "PROPOSAL",
            "negotiation": "PROPOSAL",
            "problem-resolution": "PROPOSAL",
            "problem_resolution": "PROPOSAL",
            "problem resolution": "PROPOSAL",
            "customer": "CUSTOMER",
            "won": "CUSTOMER",
            "closed": "CUSTOMER",
        }
        return mapping.get((stage or "").strip().lower(), "NEW")

    @staticmethod
    def _map_task_status(status: str) -> str:
        """Map a domain task status to a Twenty task status."""
        mapping = {
            "open": "TODO",
            "todo": "TODO",
            "in_progress": "IN_PROGRESS",
            "in-progress": "IN_PROGRESS",
            "completed": "DONE",
            "done": "DONE",
            "cancelled": "DONE",
            "canceled": "DONE",
        }
        return mapping.get((status or "").strip().lower(), "TODO")

    @staticmethod
    def _to_amount(value: Optional[float], currency: str = "USD"):
        """Convert a float amount to Twenty's {amountMicros, currencyCode}."""
        if value is None:
            return None
        return {"amountMicros": int(value * 1_000_000), "currencyCode": currency}

    @staticmethod
    def _extract_find_many_records(mcp_result: dict) -> list[dict]:
        """Extract the record list from a find_many_* MCP response."""
        seen: set[int] = set()
        preferred = (
            "result",
            "records",
            "data",
            "items",
            "results",
            "list",
            "nodes",
            "edges",
            "people",
            "companies",
            "opportunities",
            "tasks",
        )
        skip = {"content", "jsonrpc", "error", "success", "message", "id"}

        def walk(value):
            if isinstance(value, (dict, list)):
                if id(value) in seen:
                    return []
                seen.add(id(value))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if not isinstance(value, dict):
                return []
            for key in preferred:
                if key in value:
                    found = walk(value[key])
                    if found:
                        return found
            for key, item in value.items():
                if key in skip:
                    continue
                found = walk(item)
                if found:
                    return found
            return []

        return walk(mcp_result)

    def _find_existing_person_id(self, first_name: str, last_name: str) -> Optional[str]:
        """Find an existing person by first+last name (case-insensitive).

        Checks the local DB first, then Twenty CRM via MCP, to avoid
        creating duplicate person records across task runs.
        """
        fname = (first_name or "").strip().lower()
        lname = (last_name or "").strip().lower()
        for person in self.db.people.values():
            if (person.first_name or "").strip().lower() == fname and (
                person.last_name or ""
            ).strip().lower() == lname:
                return person.id
        try:
            mcp_result = self._call_mcp(
                "find_many_people",
                {"limit": 200},
            )
        except Exception:
            mcp_result = None
        for record in self._extract_find_many_records(mcp_result or {}):
            if not isinstance(record, dict) or not record.get("id"):
                continue
            record_first, record_last = self._v2_name(record)
            if record_first == fname and record_last == lname:
                return record["id"]
        try:
            mcp_result = self._call_mcp(
                "find_many_people",
                {
                    "name": {
                        "firstName": {"ilike": first_name},
                        "lastName": {"ilike": last_name},
                    },
                    "limit": 1,
                },
            )
        except Exception:
            return None
        for record in self._extract_find_many_records(mcp_result):
            if isinstance(record, dict) and record.get("id"):
                return record["id"]
        return None

    def _find_existing_company_id(self, name: str) -> Optional[str]:
        """Find an existing company by name (case-insensitive).

        Checks the local DB first, then Twenty CRM via MCP, to avoid
        creating duplicate company records across task runs.
        """
        needle = (name or "").strip().lower()
        for company in self.db.companies.values():
            if (company.name or "").strip().lower() == needle:
                return company.id
        try:
            mcp_result = self._call_mcp(
                "find_many_companies",
                {"limit": 200},
            )
        except Exception:
            mcp_result = None
        for record in self._extract_find_many_records(mcp_result or {}):
            if not isinstance(record, dict) or not record.get("id"):
                continue
            if (record.get("name") or "").strip().lower() == needle:
                return record["id"]
        try:
            mcp_result = self._call_mcp(
                "find_many_companies",
                {"name": {"ilike": name}, "limit": 1},
            )
        except Exception:
            return None
        for record in self._extract_find_many_records(mcp_result):
            if isinstance(record, dict) and record.get("id"):
                return record["id"]
        return None

    def _find_existing_opportunity_id(
        self, name: str, person_id: str
    ) -> Optional[str]:
        """Find an existing opportunity by name + person (local first, then Twenty)."""
        needle = (name or "").strip().lower()
        for opp in self.db.opportunities.values():
            if (opp.name or "").strip().lower() == needle and opp.person_id == person_id:
                return opp.id
        try:
            mcp_result = self._call_mcp(
                "find_many_opportunities",
                {"limit": 200},
            )
        except Exception:
            mcp_result = None
        for record in self._extract_find_many_records(mcp_result or {}):
            if not isinstance(record, dict) or not record.get("id"):
                continue
            if (record.get("name") or "").strip().lower() != needle:
                continue
            record_person = record.get("pointOfContactId") or record.get("personId")
            if isinstance(record_person, dict):
                record_person = record_person.get("id")
            if person_id and record_person != person_id:
                continue
            return record["id"]
        try:
            mcp_result = self._call_mcp(
                "find_many_opportunities",
                {
                    "name": {"ilike": name},
                    "pointOfContactId": {"eq": person_id},
                    "limit": 1,
                },
            )
        except Exception:
            return None
        for record in self._extract_find_many_records(mcp_result):
            if isinstance(record, dict) and record.get("id"):
                return record["id"]
        return None

    def _find_existing_task_id(self, subject: str) -> Optional[str]:
        """Find an existing task by title (local first, then Twenty)."""
        needle = (subject or "").strip().lower()
        for task in self.db.tasks.values():
            if (task.subject or "").strip().lower() == needle:
                return task.id
        try:
            mcp_result = self._call_mcp(
                "find_many_tasks",
                {"limit": 200},
            )
        except Exception:
            mcp_result = None
        for record in self._extract_find_many_records(mcp_result or {}):
            if not isinstance(record, dict) or not record.get("id"):
                continue
            title = (record.get("title") or record.get("subject") or "").strip().lower()
            if title == needle:
                return record["id"]
        try:
            mcp_result = self._call_mcp(
                "find_many_tasks",
                {"title": {"ilike": subject}, "limit": 1},
            )
        except Exception:
            return None
        for record in self._extract_find_many_records(mcp_result):
            if isinstance(record, dict) and record.get("id"):
                return record["id"]
        return None

    def _find_person_in_twenty(self, first_name: str, last_name: str) -> Optional[str]:
        """Find a person's Twenty CRM ID by first+last name (case-insensitive)."""
        try:
            mcp_result = self._call_mcp(
                "find_many_people",
                {
                    "name": {
                        "firstName": {"ilike": first_name},
                        "lastName": {"ilike": last_name},
                    },
                    "limit": 1,
                },
            )
        except Exception:
            return None
        for record in self._extract_find_many_records(mcp_result):
            if isinstance(record, dict) and record.get("id"):
                return record["id"]
        return None

    def _find_opportunities_in_twenty(self, person_id: str) -> list[dict]:
        """Find opportunities in Twenty CRM linked to a person ID."""
        try:
            mcp_result = self._call_mcp(
                "find_many_opportunities",
                {"pointOfContactId": {"eq": person_id}, "limit": 100},
            )
        except Exception:
            mcp_result = None
        opportunities = self._extract_find_many_records(mcp_result or {})
        if opportunities:
            return opportunities
        return [
            {
                "id": o.id,
                "name": o.name,
                "pointOfContactId": o.person_id,
                "companyId": o.company_id,
                "stage": o.stage,
            }
            for o in self.db.opportunities.values()
            if o.person_id == person_id
        ]

    def verify_person_exists(self, first_name: str, last_name: str) -> bool:
        """Return True if a person with the given name exists in Twenty CRM."""
        return self._find_person_in_twenty(first_name, last_name) is not None

    def verify_opportunity_exists(self, first_name: str, last_name: str) -> bool:
        """Return True if the named person has at least one opportunity in Twenty CRM."""
        person_id = self._find_person_in_twenty(first_name, last_name)
        if not person_id:
            return False
        return bool(self._find_opportunities_in_twenty(person_id))

    def verify_opportunity_stage(
        self, first_name: str, last_name: str, stage: str
    ) -> bool:
        """Return True if the named person has an opportunity at the mapped Twenty stage."""
        person_id = self._find_person_in_twenty(first_name, last_name)
        if not person_id:
            return False
        target = self._map_stage(stage)
        for opp in self._find_opportunities_in_twenty(person_id):
            if (opp.get("stage") or "").upper() == target:
                return True
        return False

    def verify_person_trust_low(
        self, first_name: str, last_name: str, max_trust: int = 25
    ) -> bool:
        """Return True if the named person has a trust level below max_trust locally."""
        fname = (first_name or "").strip().lower()
        lname = (last_name or "").strip().lower()
        for person in self.db.people.values():
            if (person.first_name or "").strip().lower() == fname and (
                person.last_name or ""
            ).strip().lower() == lname:
                return person.trust_level is not None and person.trust_level < max_trust
        return False

    # --- v2 strict verifiers ---

    @staticmethod
    def _v2_name(record: dict) -> tuple[str, str]:
        name = record.get("name") or {}
        if not isinstance(name, dict):
            name = {}
        first = record.get("firstName") or name.get("firstName") or ""
        last = record.get("lastName") or name.get("lastName") or ""
        return (first or "").strip().lower(), (last or "").strip().lower()

    def _v2_records(self, tool_name: str, arguments: dict | None = None) -> list[dict]:
        try:
            result = self._call_mcp(tool_name, arguments or {})
            return self._extract_find_many_records(result)
        except Exception as exc:
            logger.warning(f"MCP read {tool_name} failed: {exc}")
            return []

    def _v2_people(self, first_name=None, last_name=None) -> list[dict]:
        people = self._v2_records("find_many_people", {"limit": 200})
        if not people:
            people = [
                {
                    "id": p.id,
                    "firstName": p.first_name,
                    "lastName": p.last_name,
                    "name": {
                        "firstName": p.first_name,
                        "lastName": p.last_name,
                    },
                    "companyId": p.company_id,
                }
                for p in self.db.people.values()
            ]
        if first_name:
            needle = str(first_name).strip().lower()
            people = [p for p in people if self._v2_name(p)[0] == needle]
        if last_name:
            needle = str(last_name).strip().lower()
            people = [p for p in people if self._v2_name(p)[1] == needle]
        return people

    def _v2_companies(self, company_name=None) -> list[dict]:
        companies = self._v2_records("find_many_companies", {"limit": 200})
        if not companies:
            companies = [
                {"id": c.id, "name": c.name}
                for c in self.db.companies.values()
            ]
        if company_name:
            needle = str(company_name).strip().lower()
            companies = [
                c for c in companies
                if (c.get("name") or "").strip().lower() == needle
            ]
        return companies

    def _v2_opportunities(self) -> list[dict]:
        opportunities = self._v2_records(
            "find_many_opportunities", {"limit": 200}
        )
        if not opportunities:
            opportunities = [
                {
                    "id": o.id,
                    "name": o.name,
                    "pointOfContactId": o.person_id,
                    "companyId": o.company_id,
                    "stage": o.stage,
                }
                for o in self.db.opportunities.values()
            ]
        return opportunities

    def _v2_tasks(self, title_contains=None) -> list[dict]:
        tasks = self._v2_records("find_many_tasks", {"limit": 200})
        if not tasks:
            tasks = [
                {
                    "id": t.id,
                    "title": t.subject,
                    "subject": t.subject,
                    "status": t.status,
                }
                for t in self.db.tasks.values()
            ]
        if title_contains:
            needle = str(title_contains).strip().lower()
            tasks = [
                t for t in tasks
                if needle in (t.get("title") or t.get("subject") or "").strip().lower()
            ]
        return tasks

    def verify_person_named_company(
        self, first_name: str, last_name: str, company_name: str
    ) -> bool:
        people = self._v2_people(first_name, last_name)
        if len(people) != 1:
            return False
        company_id = people[0].get("companyId")
        if isinstance(company_id, dict):
            company_id = company_id.get("id")
        if not company_id:
            company = people[0].get("company") or {}
            if isinstance(company, dict):
                company_id = company.get("id")
        if not company_id:
            return False
        companies = self._v2_companies(company_name)
        return any(c.get("id") == company_id for c in companies)

    def verify_opportunity_named(
        self, first_name: str, last_name: str, name_contains: str
    ) -> bool:
        people = self._v2_people(first_name, last_name)
        if len(people) != 1:
            return False
        opportunities = self._find_opportunities_in_twenty(people[0]["id"])
        needle = (name_contains or "").strip().lower()
        return any(
            needle in (o.get("name") or "").strip().lower()
            for o in opportunities
        )

    def verify_task_named(self, title_contains: str) -> bool:
        return len(self._v2_tasks(title_contains)) >= 1

    def verify_people_total(self, count: int) -> bool:
        return len(self._v2_people()) == count

    def verify_opportunities_total(self, count: int) -> bool:
        return len(self._v2_opportunities()) == count

    def verify_tasks_total(self, count: int) -> bool:
        return len(self._v2_tasks()) == count

    def verify_person_count(self, first_name: str, last_name: str, count: int) -> bool:
        return len(self._v2_people(first_name, last_name)) == count

    def verify_opportunity_count(
        self, first_name: str, last_name: str, count: int
    ) -> bool:
        people = self._v2_people(first_name, last_name)
        if len(people) != 1:
            return False
        return len(self._find_opportunities_in_twenty(people[0]["id"])) == count

    def verify_clean_state(self) -> bool:
        return (
            len(self._v2_people()) == 0
            and len(self._v2_companies()) == 0
            and len(self._v2_opportunities()) == 0
            and len(self._v2_tasks()) == 0
        )

    def _get_person(self, person_id: str) -> Person:
        """Get person from database."""
        if person_id not in self.db.people:
            raise ValueError(f"Person {person_id} not found")
        return self.db.people[person_id]
    
    def _get_company(self, company_id: str) -> Company:
        """Get company from database."""
        if company_id not in self.db.companies:
            raise ValueError(f"Company {company_id} not found")
        return self.db.companies[company_id]
    
    def _get_new_person_id(self) -> str:
        """Generate new person ID."""
        existing_ids = set(self.db.people.keys())
        counter = 1
        while True:
            new_id = f"person_{counter}"
            if new_id not in existing_ids:
                return new_id
            counter += 1
    
    def _get_new_company_id(self) -> str:
        """Generate new company ID."""
        existing_ids = set(self.db.companies.keys())
        counter = 1
        while True:
            new_id = f"company_{counter}"
            if new_id not in existing_ids:
                return new_id
            counter += 1
    
    def _get_new_opportunity_id(self) -> str:
        """Generate new opportunity ID."""
        existing_ids = set(self.db.opportunities.keys())
        counter = 1
        while True:
            new_id = f"opportunity_{counter}"
            if new_id not in existing_ids:
                return new_id
            counter += 1
    
    def _get_new_task_id(self) -> str:
        """Generate new task ID."""
        existing_ids = set(self.db.tasks.keys())
        counter = 1
        while True:
            new_id = f"task_{counter}"
            if new_id not in existing_ids:
                return new_id
            counter += 1
    
    @is_tool(ToolType.READ)
    def find_people(self, 
                    first_name: Optional[str] = None,
                    last_name: Optional[str] = None,
                    email: Optional[str] = None,
                    company_id: Optional[str] = None,
                    limit: int = 10,
                    offset: int = 0) -> list[Person]:
        """Find people by criteria.
        
        Args:
            first_name: Filter by first name (supports partial match)
            last_name: Filter by last name (supports partial match)
            email: Filter by email
            company_id: Filter by company ID
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of matching people
        """
        results = []
        for person in self.db.people.values():
            if first_name and first_name.lower() not in person.first_name.lower():
                continue
            if last_name and last_name.lower() not in person.last_name.lower():
                continue
            if email and person.email and email.lower() not in person.email.lower():
                continue
            if company_id and person.company_id != company_id:
                continue
            results.append(person)
        
        return results[offset:offset + limit]
    
    @is_tool(ToolType.READ)
    def find_companies(self,
                       name: Optional[str] = None,
                       industry: Optional[str] = None,
                       limit: int = 10,
                       offset: int = 0) -> list[Company]:
        """Find companies by criteria.
        
        Args:
            name: Filter by company name (supports partial match)
            industry: Filter by industry
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of matching companies
        """
        results = []
        for company in self.db.companies.values():
            if name and name.lower() not in company.name.lower():
                continue
            if industry and company.industry and industry.lower() != company.industry.lower():
                continue
            results.append(company)
        
        return results[offset:offset + limit]
    
    @is_tool(ToolType.READ)
    def find_opportunities(self,
                           person_id: Optional[str] = None,
                           stage: Optional[str] = None,
                           limit: int = 10,
                           offset: int = 0) -> list[Opportunity]:
        """Find opportunities by criteria.
        
        Args:
            person_id: Filter by person ID
            stage: Filter by stage
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of matching opportunities
        """
        results = []
        for opp in self.db.opportunities.values():
            if person_id and opp.person_id != person_id:
                continue
            if stage and opp.stage != stage:
                continue
            results.append(opp)
        
        return results[offset:offset + limit]
    
    @is_tool(ToolType.READ)
    def find_tasks(self,
                   person_id: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 10,
                   offset: int = 0) -> list[Task]:
        """Find tasks by criteria.
        
        Args:
            person_id: Filter by person ID
            status: Filter by status (open, completed, cancelled)
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of matching tasks
        """
        results = []
        for task in self.db.tasks.values():
            if person_id and task.person_id != person_id:
                continue
            if status and task.status != status:
                continue
            results.append(task)
        
        return results[offset:offset + limit]
    
    @is_tool(ToolType.WRITE)
    def create_person(self,
                      first_name: str,
                      last_name: str,
                      email: Optional[str] = None,
                      phone: Optional[str] = None,
                      job_title: Optional[str] = None,
                      company_id: Optional[str] = None,
                      lead_score: int = 0,
                      relationship_stage: str = "prospect",
                      trust_level: int = 50,
                      commitment_level: int = 0) -> Person:
        """Create a new person record in Twenty CRM via MCP.
        
        Args:
            first_name: Person's first name
            last_name: Person's last name
            email: Person's email address
            phone: Person's phone number
            job_title: Person's job title
            company_id: Associated company ID
            lead_score: Initial lead score (0-100)
            relationship_stage: Initial relationship stage
            trust_level: Initial trust level (0-100)
            commitment_level: Initial commitment level (0-100)
            
        Returns:
            Created person record
        """
        # Deduplicate: reuse an existing record if one already exists.
        existing_id = self._find_existing_person_id(first_name, last_name)
        if existing_id:
            person_id = existing_id
        else:
            # Call Twenty CRM MCP
            mcp_result = self._call_mcp("create_one_person", {
                "name": {
                    "firstName": first_name,
                    "lastName": last_name
                },
                "emails": {
                    "primaryEmail": email
                } if email else {},
                "phones": {
                    "primaryPhoneNumber": phone
                } if phone else {},
                "jobTitle": job_title,
                "companyId": company_id,
                "position": "last"
            })
            
            # Extract person ID from MCP response
            # MCP returns: {"success": true, "result": {"id": "uuid"}, ...}
            if mcp_result.get("success") and "result" in mcp_result:
                person_id = mcp_result["result"].get("id")
            else:
                person_id = self._find_existing_person_id(first_name, last_name)
                if not person_id:
                    raise RuntimeError(
                        "MCP create_one_person failed: "
                        f"{mcp_result.get('message') or mcp_result}"
                    )
        
        # Create local record with MCP data
        person = Person(
            id=person_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            job_title=job_title,
            company_id=company_id,
            lead_score=lead_score,
            relationship_stage=relationship_stage,
            trust_level=trust_level,
            commitment_level=commitment_level
        )
        self.db.people[person_id] = person
        return person
    
    @is_tool(ToolType.WRITE)
    def update_person_trust_commitment(self,
                                       person_id: str,
                                       trust_level: Optional[int] = None,
                                       commitment_level: Optional[int] = None,
                                       relationship_stage: Optional[str] = None) -> Person:
        """Update trust and commitment levels for a person.
        
        This is central to the Commitment-Trust Theory framework.
        
        Args:
            person_id: Person ID to update
            trust_level: New trust level (0-100)
            commitment_level: New commitment level (0-100)
            relationship_stage: New relationship stage
            
        Returns:
            Updated person record
        """
        person = self._get_person(person_id)
        
        if trust_level is not None:
            person.trust_level = max(0, min(100, trust_level))
        if commitment_level is not None:
            person.commitment_level = max(0, min(100, commitment_level))
        if relationship_stage is not None:
            person.relationship_stage = relationship_stage
        
        return person
    
    @is_tool(ToolType.WRITE)
    def update_person_lead_score(self,
                                  person_id: str,
                                  lead_score: int) -> Person:
        """Update lead score for a person.
        
        Args:
            person_id: Person ID to update
            lead_score: New lead score (0-100)
            
        Returns:
            Updated person record
        """
        person = self._get_person(person_id)
        person.lead_score = max(0, min(100, lead_score))
        return person
    
    @is_tool(ToolType.WRITE)
    def create_company(self,
                       name: str,
                       industry: Optional[str] = None,
                       annual_revenue: Optional[float] = None,
                       employee_count: Optional[int] = None) -> Company:
        """Create a new company record in Twenty CRM via MCP.
        
        Args:
            name: Company name
            industry: Company industry
            annual_revenue: Annual revenue
            employee_count: Number of employees
            
        Returns:
            Created company record
        """
        # Deduplicate: reuse an existing record if one already exists.
        existing_id = self._find_existing_company_id(name)
        if existing_id:
            company_id = existing_id
        else:
            # Call Twenty CRM MCP
            company_args = {
                "name": name,
                "position": "last",
            }
            if annual_revenue is not None:
                company_args["annualRevenue"] = self._to_amount(annual_revenue)
            mcp_result = self._call_mcp("create_one_company", company_args)
            
            # Extract company ID from MCP response
            if mcp_result.get("success") and "result" in mcp_result:
                company_id = mcp_result["result"].get("id")
            else:
                company_id = self._find_existing_company_id(name)
                if not company_id:
                    raise RuntimeError(
                        "MCP create_one_company failed: "
                        f"{mcp_result.get('message') or mcp_result}"
                    )
        
        company = Company(
            id=company_id,
            name=name,
            industry=industry,
            annual_revenue=annual_revenue,
            employee_count=employee_count
        )
        self.db.companies[company_id] = company
        return company
    
    @is_tool(ToolType.WRITE)
    def create_opportunity(self,
                           name: str,
                           person_id: str,
                           company_id: Optional[str] = None,
                           stage: str = "NEW",
                           value: Optional[float] = None,
                           probability: float = 0.1,
                           expected_close_date: Optional[str] = None) -> Opportunity:
        """Create a new opportunity record in Twenty CRM via MCP.
        
        Args:
            name: Opportunity name
            person_id: Associated person ID
            company_id: Associated company ID (optional)
            stage: Initial stage in sales pipeline
            value: Opportunity value
            probability: Success probability (0-1)
            expected_close_date: Expected close date
            
        Returns:
            Created opportunity record
        """
        # Deduplicate: reuse an existing record if one already exists.
        existing_id = self._find_existing_opportunity_id(name, person_id)
        if existing_id:
            opportunity_id = existing_id
        else:
            # Call Twenty CRM MCP
            opp_args = {
                "name": name,
                "stage": self._map_stage(stage),
                "position": "last",
            }
            if company_id:
                opp_args["companyId"] = company_id
            if person_id:
                opp_args["pointOfContactId"] = person_id
            if value is not None:
                opp_args["amount"] = self._to_amount(value)
            if expected_close_date:
                opp_args["closeDate"] = expected_close_date
            mcp_result = self._call_mcp("create_one_opportunity", opp_args)
            
            # Extract opportunity ID from MCP response
            if mcp_result.get("success") and "result" in mcp_result:
                opportunity_id = mcp_result["result"].get("id")
            else:
                opportunity_id = self._find_existing_opportunity_id(name, person_id)
                if not opportunity_id:
                    raise RuntimeError(
                        "MCP create_one_opportunity failed: "
                        f"{mcp_result.get('message') or mcp_result}"
                    )
        
        opportunity = Opportunity(
            id=opportunity_id,
            name=name,
            person_id=person_id,
            company_id=company_id,
            stage=stage,
            value=value,
            probability=probability,
            expected_close_date=expected_close_date
        )
        self.db.opportunities[opportunity_id] = opportunity
        return opportunity
    
    @is_tool(ToolType.WRITE)
    def update_opportunity_stage(self,
                                  opportunity_id: str,
                                  stage: str) -> Opportunity:
        """Update opportunity stage in sales pipeline.
        
        Args:
            opportunity_id: Opportunity ID to update
            stage: New stage in sales pipeline
            
        Returns:
            Updated opportunity record
        """
        opportunity = self.db.opportunities[opportunity_id]
        opportunity.stage = stage
        
        # Update in MCP if available
        try:
            self._call_mcp("update_one_opportunity", {
                "id": opportunity_id,
                "stage": self._map_stage(stage)
            })
        except Exception as exc:
            raise RuntimeError(f"MCP update failed: {exc}") from exc
        
        return opportunity
    
    @is_tool(ToolType.WRITE)
    def create_task(self,
                    subject: str,
                    person_id: Optional[str] = None,
                    company_id: Optional[str] = None,
                    status: str = "open",
                    priority: Optional[str] = None) -> Task:
        """Create a new task record in Twenty CRM via MCP.
        
        Args:
            subject: Task subject/description
            person_id: Associated person ID
            company_id: Associated company ID
            status: Initial status
            priority: Task priority
            
        Returns:
            Created task record
        """
        # Deduplicate: reuse an existing record if one already exists.
        existing_id = self._find_existing_task_id(subject)
        if existing_id:
            task_id = existing_id
        else:
            # Call Twenty CRM MCP
            task_args = {
                "title": subject,
                "status": self._map_task_status(status),
                "position": "last",
            }
            mcp_result = self._call_mcp("create_one_task", task_args)
            
            # Extract task ID from MCP response
            if mcp_result.get("success") and "result" in mcp_result:
                task_id = mcp_result["result"].get("id")
            else:
                task_id = self._find_existing_task_id(subject)
                if not task_id:
                    raise RuntimeError(
                        "MCP create_one_task failed: "
                        f"{mcp_result.get('message') or mcp_result}"
                    )
        
        task = Task(
            id=task_id,
            subject=subject,
            person_id=person_id,
            company_id=company_id,
            status=status,
            priority=priority
        )
        self.db.tasks[task_id] = task
        return task
    
    @is_tool(ToolType.WRITE)
    def update_task_status(self,
                            task_id: str,
                            status: str) -> Task:
        """Update task status.
        
        Args:
            task_id: Task ID to update
            status: New status (open, completed, cancelled)
            
        Returns:
            Updated task record
        """
        task = self.db.tasks[task_id]
        task.status = status
        
        # Update in MCP if available
        try:
            self._call_mcp("update_one_task", {
                "id": task_id,
                "status": self._map_task_status(status)
            })
        except Exception as exc:
            raise RuntimeError(f"MCP update failed: {exc}") from exc
        
        return task
    
    @is_tool(ToolType.READ)
    def get_person_details(self, person_id: str) -> Person:
        """Get detailed information about a person.
        
        Args:
            person_id: Person ID to retrieve
            
        Returns:
            Person record with all details
        """
        return self._get_person(person_id)
    
    @is_tool(ToolType.READ)
    def get_company_details(self, company_id: str) -> Company:
        """Get detailed information about a company.
        
        Args:
            company_id: Company ID to retrieve
            
        Returns:
            Company record with all details
        """
        return self._get_company(company_id)
    
    @is_tool(ToolType.READ)
    def get_opportunity_details(self, opportunity_id: str) -> Opportunity:
        """Get detailed information about an opportunity.
        
        Args:
            opportunity_id: Opportunity ID to retrieve
            
        Returns:
            Opportunity record with all details
        """
        return self.db.opportunities[opportunity_id]
    
    @is_tool(ToolType.READ)
    def get_task_details(self, task_id: str) -> Task:
        """Get detailed information about a task.
        
        Args:
            task_id: Task ID to retrieve
            
        Returns:
            Task record with all details
        """
        return self.db.tasks[task_id]
    
    @is_tool(ToolType.READ)
    def get_relationship_metrics(self, person_id: str) -> dict:
        """Get relationship metrics based on Commitment-Trust Theory.
        
        Args:
            person_id: Person ID to analyze
            
        Returns:
            Dictionary with trust, commitment, and relationship metrics
        """
        person = self._get_person(person_id)
        return {
            "trust_level": person.trust_level,
            "commitment_level": person.commitment_level,
            "relationship_stage": person.relationship_stage,
            "lead_score": person.lead_score,
            "opportunity_count": len([o for o in self.db.opportunities.values() 
                                     if o.person_id == person_id]),
            "task_count": len([t for t in self.db.tasks.values() 
                              if t.person_id == person_id and t.status == "open"]),
        }
