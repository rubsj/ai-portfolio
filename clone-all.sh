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

# --- Generate VS Code workspace file with friendly display names ---
WORKSPACE_FILE="ai-portfolio.code-workspace"

# Associative array: repo folder name → VS Code display name
declare -A DISPLAY_NAMES=(
  [ai-portfolio]="Hub: AI Portfolio"
  [ai-synthetic-data-generator]="P1 Synthetic Data"
  [ai-rag-evaluation-framework]="P2 RAG Evaluation"
  [ai-contrastive-embedding-finetuning]="P3 Embedding Fine-Tuning"
  [ai-resume-coach]="P4 Resume Coach"
  [ai-shoptalk-knowledge-agent]="P5 ShopTalk RAG"
  [ai-digital-clone]="P6 Digital Clone"
  [ai-feedback-intelligence]="P7 Feedback Intelligence"
  [ai-jira-agent]="P8 Jira Agent"
  [ai-devops-assistant]="P9 DevOps Assistant"
)

echo "Generating $WORKSPACE_FILE..."

# Build the JSON folders array from repos that actually exist on disk
FOLDERS=""
FIRST=true
for repo in "${REPOS[@]}"; do
  if [ -d "$repo" ]; then
    name="${DISPLAY_NAMES[$repo]:-$repo}"
    if [ "$FIRST" = true ]; then
      FIRST=false
    else
      FOLDERS+=","
    fi
    FOLDERS+=$'\n\t\t'"{ \"path\": \"./$repo\", \"name\": \"$name\" }"
  fi
done

cat > "$WORKSPACE_FILE" <<EOF
{
	"folders": [$FOLDERS
	],
	"settings": {}
}
EOF

echo ""
echo "Workspace file created at $(pwd)/$WORKSPACE_FILE"
echo "Open it with: code $WORKSPACE_FILE"
