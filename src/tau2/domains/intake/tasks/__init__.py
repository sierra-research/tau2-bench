# Copyright Sierra
"""Entity banks and name-bank ingestion for the intake domain.

What remains of the v3 task pipeline after the v4 rip (design doc §8): the
value banks (:mod:`tau2.domains.intake.tasks.banks`) and the census/SSA
person-name bank builder behind ``tau2 intake-names``
(:mod:`tau2.domains.intake.tasks.name_banks`). The v4 generator, enumeration,
and freeze are rebuilt in Phase C (design doc §9).
"""
