from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class GovernmentAdapter(ABC):
    """Abstract interface for all municipal government integrations."""

    @abstractmethod
    async def submit_complaint(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches complaint packet to municipal authority.
        Returns: Dict containing success boolean, tracking ticket ID, and dispatch details.
        """
        pass

    @abstractmethod
    async def check_status(self, ticket_id: str) -> Dict[str, Any]:
        """
        Checks latest complaint status from the municipal grievance system.
        """
        pass
