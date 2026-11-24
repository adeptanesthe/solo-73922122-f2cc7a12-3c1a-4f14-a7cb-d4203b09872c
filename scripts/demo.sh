#!/usr/bin/env bash
# Demo: one film, several channel video ids sharing a single shot index,
# each with its own restriction orders.  Ends with the resident-index
# comparison (shared catalog vs. one full index copy per video id).
# Prereq: service listening on ${BASE:-http://localhost:8000}
#   docker compose up --build        # or: uvicorn app.main:app --port 8000
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"

req() { # method path [json-body] -> body + status line
  local method="$1" path="$2" body="${3:-}"
  echo
  local shown="${body:0:100}"
  if [ "${#body}" -gt 100 ]; then shown="$shown ... (${#body} bytes)"; fi
  echo "### $method $path ${shown:+ $shown}"
  if [ -n "$body" ]; then
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path" \
      -H 'Content-Type: application/json' --data "$body"
  else
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path"
  fi
}

# post_file path file -> body + status line (for the big shot payload)
post_file() {
  echo
  echo "### POST $1  <$(basename "$2")>"
  curl -s -w $'\nHTTP %{http_code}\n' -X POST "$BASE$1" \
    -H 'Content-Type: application/json' --data @"$2"
}

echo "=== 1. film 'demo' imported under 3 channel video ids (same 10 shots) ==="
for vid in channel-a channel-b channel-c; do
  post_file "/videos/$vid/shots" "$HERE/demo_shots.json"
done

echo
echo "=== 2. storage after 3 imports: ONE shared catalog, 3 tiny overlays ==="
curl -s -w $'\nHTTP %{http_code}\n' "$BASE/storage"

echo
echo "=== 3. channel-a: order lic-A hides [2,6) -> 6 playable ==="
req POST /videos/channel-a/restrictions '{"order_id":"lic-A","start":2,"end":6}'

echo
echo "=== 4. channel-b: same order id, DIFFERENT range [0,4) -> only b is affected ==="
req POST /videos/channel-b/restrictions '{"order_id":"lic-A","start":0,"end":4}'

echo
echo "=== 5. channel-c still sees all 10 shots (orders are per video id) ==="
req GET /videos/channel-c/retrieval-vector

echo
echo "=== 6. re-send lic-A on channel-a with the SAME range: idempotent -> 6 playable ==="
req POST /videos/channel-a/restrictions '{"order_id":"lic-A","start":2,"end":6}'

echo
echo "=== 7. re-use lic-A on channel-a with a DIFFERENT range: 409 conflict ==="
req POST /videos/channel-a/restrictions '{"order_id":"lic-A","start":3,"end":7}'

echo
echo "=== 8. revoke lic-A on channel-a: all 10 playable again ==="
req DELETE /videos/channel-a/restrictions/lic-A
req DELETE /videos/channel-b/restrictions/lic-A

echo
echo "=== 9. channel-a: order takedown hides [0,10) -> withdrawn marker ==="
req POST /videos/channel-a/restrictions '{"order_id":"takedown","start":0,"end":10}'
req DELETE /videos/channel-a/restrictions/takedown

echo
echo "=== 10. channel-c switches to a different film -> 10-shot catalog survives (a, b) ==="
BODY='{"shots":[{"feature":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6],"score":0.0}]}'
req POST /videos/channel-c/shots "$BODY"
curl -s "$BASE/storage" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ("catalog_count","video_count","retained_bytes_total","naive_bytes_total","saved_bytes")}))'

echo
echo "=== 11. channel-b leaves too: still 2 video ids on the 10-shot catalog ==="
req POST /videos/channel-b/shots "$BODY"
curl -s "$BASE/storage" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ("catalog_count","video_count","retained_bytes_total","naive_bytes_total","saved_bytes")}))'

echo
echo "=== 12. last video id (channel-a) leaves -> old 10-shot index is freed ==="
req POST /videos/channel-a/shots "$BODY"

echo
echo "=== 13. final storage: one 1-shot catalog shared by 3 video ids ==="
curl -s -w $'\nHTTP %{http_code}\n' "$BASE/storage"

echo
echo "=== 14. revoking an unknown order: 404 ==="
req DELETE /videos/channel-a/restrictions/no-such-order
