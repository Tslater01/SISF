# src/sisf/utils/policy_store.py
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
        # The separate '_active_policy_ids' set is now REMOVED.
        self._lock = threading.Lock()
        print("Adaptive Policy Store (APS) initialized (in-memory).")

    def add_policy(self, policy: Policy, activate: bool = True):
        """Adds a new policy to the store."""
        with self._lock:
            if policy.id in self._policies:
                print(f"APS: Warning - Overwriting policy {policy.id}")
            # --- IMPROVEMENT: Store active state on the object itself ---
            policy.is_active = activate
            self._policies[policy.id] = policy
            print(f"APS: Added policy {policy.id}. Active: {activate}")

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Retrieves a single policy by its ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_active_policies(self) -> List[Policy]:
        """Returns a list of all currently active policies."""
        with self._lock:
            # --- IMPROVEMENT: Filter based on the object's own state ---
            return [p for p in self._policies.values() if p.is_active]
            
    def get_all_policies(self) -> List[Policy]:
        """Returns a list of all policies, active or inactive."""
        with self._lock:
            return list(self._policies.values())

    def toggle_policy(self, policy_id: str, active: bool) -> bool:
        """Activates or deactivates a policy by updating its state."""
        with self._lock:
            policy = self._policies.get(policy_id)
            if not policy:
                return False
            # --- IMPROVEMENT: Update the state directly on the object ---
            policy.is_active = active
            print(f"APS: Set policy {policy_id} active status to {active}")
            return True