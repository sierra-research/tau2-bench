#!/bin/bash

# Stage all changes (deletions and new files)
git add -A

# Commit with a single line message
git commit -m "refactor: move adversarial evaluation code to src/experiments/tau-adversarial"

# Push to remote
git push
