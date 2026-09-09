#!/usr/bin/env bash
set -euo pipefail

readonly MAX_LINES="${MAX_LINES:-500}"
readonly BASE_REF="${1:-}"
readonly HEAD_REF="${2:-HEAD}"
readonly LEGACY_FILE="${LEGACY_FILE:-.github/loc-500-legacy.txt}"

if [[ -z "$BASE_REF" ]]; then
  echo "Usage: $0 <base-ref> [head-ref]" >&2
  exit 2
fi

git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null
git rev-parse --verify "${HEAD_REF}^{commit}" >/dev/null

checked=0
violations=0

while IFS= read -r -d '' path; do
  case "$path" in
    *.py|*.sh|*.yml|*.yaml|*.toml|*.sql|*.html|*.css|*.js|*.ts)
      ;;
    *)
      continue
      ;;
  esac

  checked=$((checked + 1))
  lines=$(git show "${HEAD_REF}:$path" | awk 'END { print NR }')

  if (( lines <= MAX_LINES )); then
    continue
  fi

  if [[ -f "$LEGACY_FILE" ]] &&
    grep -Fqx -- "$path" "$LEGACY_FILE" &&
    git cat-file -e "${BASE_REF}:$path" 2>/dev/null; then
    base_lines=$(git show "${BASE_REF}:$path" | awk 'END { print NR }')
    if (( base_lines > MAX_LINES )); then
      echo "::warning file=$path::Legacy LOC exception: $lines lines (base: $base_lines, limit: $MAX_LINES). Refactor before removing it from $LEGACY_FILE."
      continue
    fi
  fi

  echo "::error file=$path::File has $lines lines; the maximum is $MAX_LINES. Split it into smaller modules."
  violations=$((violations + 1))
done < <(git diff --name-only --diff-filter=ACMR -z "$BASE_REF" "$HEAD_REF")

echo "Checked $checked changed code files against the $MAX_LINES-line limit."

if (( violations > 0 )); then
  echo "$violations file(s) exceed the limit without a legacy exception." >&2
  exit 1
fi
