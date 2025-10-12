# sisf/utils/policy_store.py
"""
Implements the Adaptive Policy Store (APS).
"""
import threading
from typing import List, Dict, Optional
from sisf.schemas.policies import Policy

class PolicyStore:
    """A thread-safe, in-memory store for safety policies."""
    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._active_policy_ids: set[str] = set()
        self._lock = threading.Lock()
        print("Adaptive Policy Store (APS) initialized (in-memory).")

    def add_policy(self, policy: Policy, activate: bool = True):
        """Adds a new policy to the store."""
        with self._lock:
            if policy.id in self._policies:
                print(f"APS: Warning - Overwriting policy {policy.id}")
            self._policies[policy.id] = policy
            if activate:
                self._active_policy_ids.add(policy.id)
            print(f"APS: Added policy {policy.id}. Active: {activate}")

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Retrieves a single policy by its ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_active_policies(self) -> List[Policy]:
        """Returns a list of all currently active policies."""
        with self._lock: # <-- FIX: was self.lock
            return [self._policies[pid] for pid in self._active_policy_ids if pid in self._policies]
            
    def get_all_policies(self) -> List[Policy]:
        """Returns a list of all policies, active or inactive."""
        with self._lock:
            return list(self._policies.values())

    def toggle_policy(self, policy_id: str, active: bool) -> bool:
        """Activates or deactivates a policy. (For HITL Oversight)"""
        with self._lock:
            if policy_id not in self._policies:
                return False
            if active:
                self._active_policy_ids.add(policy_id)
            else:
                self._active_policy_ids.discard(policy_id)
            print(f"APS: Set policy {policy_id} active status to {active}")
            return True