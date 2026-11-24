# Load config.env as defaults; variables already set in the environment win.
while IFS='=' read -r _k _v; do
  [[ -z "$_k" || "$_k" == \#* ]] && continue
  [ -z "${!_k:-}" ] && export "$_k=$_v"
done < "$(dirname "${BASH_SOURCE[0]}")/../config.env"
