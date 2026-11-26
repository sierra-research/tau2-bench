### openai_Qwen3.5-9B-UD-Q8_K_XL.gguf

| Task | Scenario | Required CRM outcome | Success | pass^1 | pass^4 | pass^8 | Failed trials |
|---|---|---|---:|---:|---:|---:|---|
| T1 | Skeptical owner, new lead | Person linked to TechStart; opportunity 'TechStart CRM Evaluation' | 6/8 | 75.0% | 21.43% | 0.00% | 2 user_stop |
| T2 | Stubborn buyer, trust | Person linked to Enterprise; task 'Trust-building follow-up' | 1/8 | 12.5% | 0.00% | 0.00% | 5 user_stop, 2 max_steps |
| T3 | Analytical manager, pilot | Person linked to Growth; 'Growth Pilot' at SCREENING | 4/8 | 50.0% | 1.43% | 0.00% | 4 user_stop |
| T4 | Evidence-seeking buyer, trial | Person linked to ScaleUp; 'ScaleUp Trial' | 8/8 | 100.0% | 100.00% | 100.00% | none |
| T5 | Cautious tester, pilot + follow-up | Person linked to Midway; 'Midway Small Pilot'; 'Follow up in 7 days' | 8/8 | 100.0% | 100.00% | 100.00% | none |
| T6 | Strategic buyer, partnership | Person linked to FutureTech; 'FutureTech Strategic Partnership' at MEETING | 5/8 | 62.5% | 7.14% | 0.00% | 3 user_stop |
| T7 | Challenged customer, complaint | Move 'Innovate Onboarding' to PROPOSAL; 'Complaint follow-up' | 8/8 | 100.0% | 100.00% | 100.00% | none |
| T8 | Advocate, referral | 'Success LLC Referral' for existing Alex Thompson | 7/8 | 87.5% | 50.00% | 0.00% | 1 user_stop |
| T9 | Loyal customer, renewal | 'Enterprise Co Renewal' for existing Rachel Brown | 4/8 | 50.0% | 1.43% | 0.00% | 4 user_stop |
| T10 | At-risk customer, recovery | 'Transition Corp Recovery' + 'Recovery follow-up' | 3/8 | 37.5% | 0.00% | 0.00% | 5 user_stop |
| **All tasks** | | | **54/80** | **67.5%** | **38.14%** | **30.00%** | |

Model-level 95% task-cluster bootstrap CIs:
- pass^1: 67.50%  CI [48.75, 85.00]
- pass^4: 38.14%  CI [13.00, 65.71]
- pass^8: 30.00%  CI [0.00, 60.00]
