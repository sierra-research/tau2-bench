### openai_gpt-oss-20b-UD-Q4_K_XL.gguf

| Task | Scenario | Required CRM outcome | Success | pass^1 | pass^4 | pass^8 | Failed trials |
|---|---|---|---:|---:|---:|---:|---|
| T1 | Skeptical owner, new lead | Person linked to TechStart; opportunity 'TechStart CRM Evaluation' | 0/8 | 0.0% | 0.00% | 0.00% | 8 user_stop |
| T2 | Stubborn buyer, trust | Person linked to Enterprise; task 'Trust-building follow-up' | 6/8 | 75.0% | 21.43% | 0.00% | 2 user_stop |
| T3 | Analytical manager, pilot | Person linked to Growth; 'Growth Pilot' at SCREENING | 7/8 | 87.5% | 50.00% | 0.00% | 1 user_stop |
| T4 | Evidence-seeking buyer, trial | Person linked to ScaleUp; 'ScaleUp Trial' | 0/8 | 0.0% | 0.00% | 0.00% | 4 too_many_errors, 4 user_stop |
| T5 | Cautious tester, pilot + follow-up | Person linked to Midway; 'Midway Small Pilot'; 'Follow up in 7 days' | 6/8 | 75.0% | 21.43% | 0.00% | 2 user_stop |
| T6 | Strategic buyer, partnership | Person linked to FutureTech; 'FutureTech Strategic Partnership' at MEETING | 7/8 | 87.5% | 50.00% | 0.00% | 1 user_stop |
| T7 | Challenged customer, complaint | Move 'Innovate Onboarding' to PROPOSAL; 'Complaint follow-up' | 5/8 | 62.5% | 7.14% | 0.00% | 3 user_stop |
| T8 | Advocate, referral | 'Success LLC Referral' for existing Alex Thompson | 8/8 | 100.0% | 100.00% | 100.00% | none |
| T9 | Loyal customer, renewal | 'Enterprise Co Renewal' for existing Rachel Brown | 7/8 | 87.5% | 50.00% | 0.00% | 1 user_stop |
| T10 | At-risk customer, recovery | 'Transition Corp Recovery' + 'Recovery follow-up' | 6/8 | 75.0% | 21.43% | 0.00% | 2 user_stop |
| **All tasks** | | | **52/80** | **65.0%** | **32.14%** | **10.00%** | |

Model-level 95% task-cluster bootstrap CIs:
- pass^1: 65.00%  CI [42.50, 83.75]
- pass^4: 32.14%  CI [15.71, 51.43]
- pass^8: 10.00%  CI [0.00, 30.00]
