# TECHNICAL SUPPORT MANUAL - AI OPTIMIZED

## DIAGNOSTIC TOOLS REFERENCE

### DIAGNOSTIC ACTIONS (Information Only)
| Tool | Purpose | Key Outputs |
|------|---------|-------------|
| `check_status_bar()` | Quick service overview | Signal strength, airplane mode, data status |
| `check_network_status()` | Detailed connectivity | Signal bars (none/poor/fair/good/excellent) |
| `check_sim_status()` | SIM card health | Active/Missing/Locked |
| `check_network_mode_preference()` | Network type | 5G/4G/3G/2G preference |
| `check_data_restriction_status()` | Data limiters | Data saver on/off |
| `check_apn_settings()` | Network config | APN name, MMSC URL status |
| `check_wifi_status()` | WiFi connection | Connected network, signal |
| `check_wifi_calling_status()` | WiFi calling | Enabled/disabled |
| `check_vpn_status()` | VPN connection | Active/inactive |
| `check_app_permissions(app_name)` | App access rights | List of granted permissions |
| `run_speed_test()` | Data speed | no connection/very poor/poor/fair/good/excellent |
| `can_send_mms()` | MMS capability | Working/not working |

### FIX ACTIONS (Make Changes)
| Tool | Action | When to Use |
|------|--------|-------------|
| `toggle_airplane_mode()` | ON/OFF airplane mode | If airplane mode is stuck |
| `reseat_sim_card()` | Remove/reinsert SIM | If SIM shows "Missing" |
| `toggle_data()` | Enable/disable mobile data | If data is turned off |
| `toggle_roaming()` | Enable/disable roaming | When traveling abroad |
| `toggle_data_saver_mode()` | Turn off data restrictions | If speed is slow |
| `set_network_mode_preference(mode)` | Change network type | Use "4g_5g_preferred" for best speed |
| `reset_apn_settings()` | Fix network config | If APN or MMSC incorrect |
| `toggle_wifi_calling()` | Turn off WiFi calling | If it conflicts with MMS |
| `disconnect_vpn()` | Disable VPN | If VPN slows connection |
| `grant_app_permission(app, permission)` | Give app access | For messaging: "storage" and "sms" |
| `reboot_device()` | Restart phone | After APN changes, general fixes |

---

## TROUBLESHOOTING WORKFLOWS

## PROBLEM: NO SERVICE / NO SIGNAL

### STEP-BY-STEP RESOLUTION:

**1. QUICK CHECK**
```
Action: Ask user to check_status_bar()
Result Analysis:
├── Shows signal bars → NOT a service issue, check other problems
└── Shows "No Signal" or "Airplane Mode" → Continue to Step 2
```

**2. AIRPLANE MODE CHECK**
```
Action: check_network_status()
Result Analysis:
├── Airplane Mode ON → User: toggle_airplane_mode() → Recheck status bar
└── Airplane Mode OFF → Continue to Step 3
```

**3. SIM CARD CHECK**
```
Action: check_sim_status()
Result Analysis:
├── SIM "Missing" → User: reseat_sim_card() → Recheck status bar
├── SIM "Locked" → TRANSFER TO HUMAN (PUK unlock needed)
└── SIM "Active" → Continue to Step 4
```

**4. NETWORK SETTINGS CHECK**
```
Action: check_apn_settings()
Result Analysis:
├── APN "Incorrect" → User: reset_apn_settings() + reboot_device() → Recheck
└── APN "Correct" → Continue to Step 5
```

**5. ACCOUNT STATUS CHECK**
```
Action: Check if line is suspended
Result Analysis:
├── Suspended for bills → Follow bill payment workflow
├── Suspended for contract → TRANSFER TO HUMAN
└── Active → TRANSFER TO HUMAN (hardware issue)
```

---

## PROBLEM: NO INTERNET / SLOW DATA

### PHASE 1: NO INTERNET CONNECTION

**1. SPEED TEST**
```
Action: User runs speed_test()
Result Analysis:
├── "no connection" → Continue troubleshooting
├── "very poor" to "good" → Skip to PHASE 2 (Slow Data)  
└── "excellent" → NOT a data issue
```

**2. SERVICE PREREQUISITE**
```
Must have cellular service first
├── No service → Follow "NO SERVICE" workflow completely
└── Service OK → Continue to Step 3
```

**3. BASIC DATA SETTINGS**
```
Action: check_network_status()
Mobile Data Status:
├── Disabled → User: toggle_data() → Recheck speed test
└── Enabled → Continue to Step 4
```

**4. ROAMING CHECK (If User Traveling)**
```
Ask: "Are you outside your usual service area?"
├── NO → Continue to Step 5
└── YES → Check roaming:
    ├── Device roaming OFF → User: toggle_roaming()
    ├── Account roaming disabled → Enable roaming (free)
    └── Both enabled → Continue to Step 5
```

**5. DATA LIMIT CHECK**
```
Action: Check user's data usage vs plan limit
Result Analysis:
├── Usage exceeded → Offer data refueling or plan change
└── Within limit → TRANSFER TO HUMAN
```

### PHASE 2: SLOW DATA SPEED

**1. DATA RESTRICTIONS**
```
Action: check_data_restriction_status()
Data Saver Status:
├── ON → User: toggle_data_saver_mode() → Recheck speed
└── OFF → Continue to Step 2
```

**2. NETWORK PREFERENCE**
```
Action: check_network_mode_preference()
Network Setting:
├── "2G" or "3G" → User: set_network_mode_preference("4g_5g_preferred")
└── Already optimal → Continue to Step 3
```

**3. VPN INTERFERENCE**
```
Action: check_vpn_status()
VPN Status:
├── Active → User: disconnect_vpn() → Recheck speed
└── Inactive → TRANSFER TO HUMAN
```

---

## PROBLEM: CANNOT SEND PICTURES (MMS)

### PREREQUISITES CHECK (MUST WORK FIRST):
1. **Cellular Service** → If NO: Complete service troubleshooting
2. **Mobile Data** → If NO: Complete data troubleshooting (any speed OK)

### MMS-SPECIFIC STEPS:

**1. NETWORK TYPE CHECK**
```
Action: check_network_status()
Network Type:
├── "2G" → User: set_network_mode_preference("4g_5g_preferred")
└── "3G" or higher → Continue to Step 2
```

**2. WIFI CALLING CONFLICT**
```
Action: check_wifi_calling_status()
WiFi Calling:
├── ON → User: toggle_wifi_calling() → Test MMS
└── OFF → Continue to Step 3
```

**3. APP PERMISSIONS**
```
Action: check_app_permissions("messaging")
Missing Permissions:
├── No "storage" → User: grant_app_permission("messaging", "storage")
├── No "sms" → User: grant_app_permission("messaging", "sms")  
└── Both present → Continue to Step 4
```

**4. MMSC CONFIGURATION**
```
Action: check_apn_settings()
MMSC URL:
├── Missing → User: reset_apn_settings() + reboot_device()
└── Present → TRANSFER TO HUMAN
```

---

## COMMON ISSUE PATTERNS

### PATTERN: Everything Works Except One Feature
```
Service ✓ + Data ✓ + MMS ✗ → Check MMS-specific settings
Service ✓ + Data ✗ + MMS ✗ → Fix data first
Service ✗ + Data ✗ + MMS ✗ → Fix service first
```

### PATTERN: Intermittent Issues
```
Works sometimes → Check:
├── Data usage near limit → Offer refueling
├── Roaming on/off → Check travel status
└── Network congestion → TRANSFER TO HUMAN
```

### PATTERN: After Travel/Location Change
```
Priority Check Order:
1. Roaming settings (device + account)
2. Network mode preference  
3. APN settings reset
```

---

## CRITICAL SUCCESS FACTORS

### ALWAYS VERIFY FIXES:
- After each step, retest the original problem
- Don't assume a setting change worked
- Use appropriate test (status bar, speed test, MMS test)

### ESCALATION TRIGGERS:
1. SIM locked with PUK
2. All troubleshooting steps exhausted
3. User has active service but carrier-side issues suspected
4. Hardware failure indicators (multiple simultaneous problems)

### COMMON AI AGENT MISTAKES TO AVOID:
1. Skipping prerequisite checks (service before data)
2. Not retesting after each fix attempt
3. Making multiple changes simultaneously
4. Assuming user knows technical terms
5. Not confirming the fix actually resolved the original problem