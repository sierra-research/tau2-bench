#!/bin/bash
#
# τ²-Adv Bench: Full Evaluation Script
#
# Runs comprehensive adversarial evaluation across:
# - 3 open-source models (GPT-OSS-120B, Nemotron-3-Nano-30B, MiMo-V2-Flash)
# - 3 domains (airline, retail, telecom)
# - 5 attack strategies
# - 3 sophistication levels (0.0, 0.5, 1.0)
#
# Usage:
#   ./run_full_evaluation.sh
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/openrouter_eval/run_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/evaluation.log"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   τ²-Adv Bench: Full Adversarial Evaluation${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY environment variable not set${NC}"
    echo "Set it with: export OPENROUTER_API_KEY=your_key_here"
    exit 1
fi

echo -e "${GREEN}✓ OpenRouter API key found${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory: ${OUTPUT_DIR}${NC}"

# Log start time
echo "Evaluation started at: $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the Python evaluation script
echo -e "${YELLOW}Starting evaluation...${NC}"
echo ""

python run_openrouter_eval.py \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    2>&1 | tee -a "$LOG_FILE"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo -e "${GREEN}============================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}   Evaluation Complete!${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}============================================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Results saved to: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # List output files
    echo "Output files:" | tee -a "$LOG_FILE"
    ls -la "$OUTPUT_DIR" | tee -a "$LOG_FILE"
else
    echo -e "${RED}Evaluation failed!${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

echo ""
echo "Evaluation finished at: $(date)" | tee -a "$LOG_FILE"
