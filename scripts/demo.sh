#!/usr/bin/env bash
# Demo: overlapping restriction orders and their revocation.
# Prereq: service listening on ${BASE:-http://localhost:8000}
#   docker compose up --build        # or: uvicorn app.main:app --port 8000
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"

req() { # method path [json-body] -> body + status line
  local method="$1" path="$2" body="${3:-}"
  echo
  echo "### $method $path ${body:+ $body}"
  if [ -n "$body" ]; then
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path" \
      -H 'Content-Type: application/json' --data "$body"
  else
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path"
  fi
}

echo "=== 1. import 10 shots (all playable) ==="
curl -s -w $'\nHTTP %{http_code}\n' -X POST "$BASE/videos/demo/shots" \
  -H 'Content-Type: application/json' --data @"$HERE/demo_shots.json"

echo
echo "=== 2. order lic-A hides [2,6) -> 6 playable ==="
req POST /videos/demo/restrictions '{"order_id":"lic-A","start":2,"end":6}'

echo
echo "=== 3. order lic-B hides [4,8) (overlaps A on [4,6)) -> 4 playable ==="
req POST /videos/demo/restrictions '{"order_id":"lic-B","start":4,"end":8}'

echo
echo "=== 4. re-send lic-A with the SAME range: idempotent, nothing changes ==="
req POST /videos/demo/restrictions '{"order_id":"lic-A","start":2,"end":6}'

echo
echo "=== 5. re-use lic-A with a DIFFERENT range: 409 conflict ==="
req POST /videos/demo/restrictions '{"order_id":"lic-A","start":3,"end":7}'

echo
echo "=== 6. revoke lic-A: [2,4) playable again, [4,8) STILL hidden by lic-B ==="
req DELETE /videos/demo/restrictions/lic-A

echo
echo "=== 7. revoke lic-B: fully recovered, vector equals the import-time one ==="
req DELETE /videos/demo/restrictions/lic-B

echo
echo "=== 8. order takedown hides [0,10): every shot hidden -> withdrawn marker ==="
req POST /videos/demo/restrictions '{"order_id":"takedown","start":0,"end":10}'

echo
echo "=== 9. revoke takedown: index recovered ==="
req DELETE /videos/demo/restrictions/takedown

echo
echo "=== 10. revoking an unknown order: 404 ==="
req DELETE /videos/demo/restrictions/no-such-order
