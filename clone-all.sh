#!/usr/bin/env bash
# Clone all AI Portfolio repos as siblings in the current directory.
# Usage: ./clone-all.sh [parent-dir]
#   parent-dir defaults to current directory

set -euo pipefail

PARENT="${1:-.}"
mkdir -p "$PARENT"
cd "$PARENT"

REPOS=(
  ai-portfolio
  ai-synthetic-data-generator
  ai-rag-evaluation-framework
  ai-contrastive-embedding-finetuning
  ai-resume-coach
  ai-shoptalk-knowledge-agent
  ai-digital-clone
  ai-feedback-intelligence
  ai-jira-agent
  ai-devops-assistant
)

for repo in "${REPOS[@]}"; do
  if [ -d "$repo" ]; then
    echo "Skipping $repo (already exists)"
  else
    echo "Cloning $repo..."
    git clone "https://github.com/rubsj/${repo}.git"
  fi
done

# Copy workspace file to parent directory for multi-root development
if [ -f "ai-portfolio/ai-portfolio.code-workspace" ]; then
  cp ai-portfolio/ai-portfolio.code-workspace .
  echo ""
  echo "Workspace file copied to $(pwd)/ai-portfolio.code-workspace"
  echo "Open it with: code ai-portfolio.code-workspace"
fi

echo ""
echo "All repos cloned. Open the workspace file in VS Code to get started."
