# Copyright Sierra
"""Audio delivery judge: fidelity + intonation, a perceptual-quality axis.

Listens to the agent's synthesized speech (unlike the text-only nativeness judge)
and scores how it is *rendered*. Public entrypoint: ``evaluate_delivery`` (in
``tau2.judges.delivery.harness``).
"""
