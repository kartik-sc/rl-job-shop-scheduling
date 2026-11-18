"""Threshold + SPT: maintain when the deciding machine's risk exceeds tau, otherwise SPT."""

from backend.baselines.dispatch import shortest_processing_time
from backend.env.simulator import Action, Decision, Maintain, Policy, Wait


def threshold_spt(tau: float) -> Policy:
    """Maintain if allowed and ``risk > tau``; else the shortest candidate; else wait.

    ``tau = 1.0`` never maintains, since risk never exceeds 1.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must lie in [0, 1], got {tau}")

    def policy(decision: Decision) -> Action:
        if decision.can_maintain and decision.risk is not None and decision.risk > tau:
            return Maintain()
        if decision.candidates:
            return shortest_processing_time(decision)
        return Wait()

    return policy
