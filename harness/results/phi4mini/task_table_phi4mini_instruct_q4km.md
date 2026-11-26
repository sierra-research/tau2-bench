### openai_Phi-4-mini-instruct-Q4_K_M.gguf

| Task | Scenario | Required CRM outcome | Success | pass^1 | pass^4 | pass^8 | Failed trials |
|---|---|---|---:|---:|---:|---:|---|
| T1 | Skeptical owner, new lead | Person linked to TechStart; opportunity 'TechStart CRM Evaluation' | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T2 | Stubborn buyer, trust | Person linked to Enterprise; task 'Trust-building follow-up' | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T3 | Analytical manager, pilot | Person linked to Growth; 'Growth Pilot' at SCREENING | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T4 | Evidence-seeking buyer, trial | Person linked to ScaleUp; 'ScaleUp Trial' | 0/8 | 0.0% | 0.00% | 0.00% | 7 max_steps, 1 user_stop |
| T5 | Cautious tester, pilot + follow-up | Person linked to Midway; 'Midway Small Pilot'; 'Follow up in 7 days' | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T6 | Strategic buyer, partnership | Person linked to FutureTech; 'FutureTech Strategic Partnership' at MEETING | 0/8 | 0.0% | 0.00% | 0.00% | 7 user_stop, 1 max_steps |
| T7 | Challenged customer, complaint | Move 'Innovate Onboarding' to PROPOSAL; 'Complaint follow-up' | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T8 | Advocate, referral | 'Success LLC Referral' for existing Alex Thompson | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T9 | Loyal customer, renewal | 'Enterprise Co Renewal' for existing Rachel Brown | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T10 | At-risk customer, recovery | 'Transition Corp Recovery' + 'Recovery follow-up' | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| **All tasks** | | | **0/80** | **0.0%** | **0.00%** | **0.00%** | |

Model-level 95% task-cluster bootstrap CIs:
- pass^1: 0.00%  CI [0.00, 0.00]
- pass^4: 0.00%  CI [0.00, 0.00]
- pass^8: 0.00%  CI [0.00, 0.00]
