# Frozen-evidence audit

- **PASS — paper_cell_roster**: found 12 agent-directed and 12 scaffolded paper cells
- **PASS — paper_call_count**: agent-directed=2400; scaffolded=2400; total=4800
- **PASS — paper_regular_pass_at_1**: all eight regular-realization rates match Figure 2
- **PASS — paper_agent_directed_pass3**: all four crossed-realization scores match Figure 2
- **PASS — paper_scaffolded_pass3**: all four crossed-realization scores match Figure 2
- **PASS — text_control**: cells=1; rows=200; passes=200
- **PASS — paper_speech_judgments**: all four utterance-level speech-fidelity rows reproduce from exported judgments
- **PASS — deterministic_complication_assignments**: checked=3200; mismatches=0
- **PASS — recoverable_composition_and_protocol_results**: two/three-field, joint-submit, and verify/retry rows reproduce exactly
- **PASS — behavioral_capture_verification_repair**: initial capture, verification, and repair numerators reproduce exactly
- **PASS — pooled_repair_pathways**: 10/98, 32/123, 7/29, and 34/97 reproduce from transcript events
- **PASS — noise_adaptivity_effort**: only Grok increases verification effort under noise
- **PASS — strategy_difficulty_and_duration**: easy/hard success and 21-28 second scaffold costs reproduce exactly
- **PASS — entity_robust_results**: all ten entity-bank robust-success rows reproduce exactly
- **PASS — realism_assignment_ledger**: clean=253/498; assigned=674/1302
- **PASS — strict_speech_validation**: P/R/F1=0.7317/1.0000/0.8451
- **PASS — lenient_speech_validation**: P/R/F1=0.8537/0.7297/0.7868
