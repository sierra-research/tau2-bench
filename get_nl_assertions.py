"""
Output:
```bash
Counter({"{'note': 'No nl_assertions to evaluate'}": 9868, 'None': 964})
```
"""
from pathlib import Path
import json
from collections import Counter
from tqdm.auto import tqdm


all_nl = Counter()
for result_file in tqdm(Path("data/tau2/results/final").glob("*.json"), desc="Reading paths"):
    data = json.loads(result_file.read_text())
    for simulation in data["simulations"]:
        nl = simulation["reward_info"]["info"].get("nl")
        all_nl[str(nl)] += 1

print(all_nl)
