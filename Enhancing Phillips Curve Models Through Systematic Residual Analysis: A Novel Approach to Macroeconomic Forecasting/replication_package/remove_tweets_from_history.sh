#!/bin/bash
# Script to remove tweets.js from git history

echo "Removing tweets.js from git history..."
echo "WARNING: This will rewrite git history!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Navigate to repository root
cd ../..

# Use filter-branch to remove the file from all commits
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch "Enhancing Phillips Curve Models Through Systematic Residual Analysis: A Novel Approach to Macroeconomic Forecasting/replication_package/tweets.js"' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up refs
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Force garbage collection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! tweets.js has been removed from history."
echo "You'll need to force push: git push origin main --force"