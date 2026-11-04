#!/bin/bash
set -euo pipefail
test ! -e inputs.sha256 || { echo 'Already sealed; use a new run directory' >&2; exit 1; }
test "$(git -C source rev-parse HEAD)" = "$(cat SOURCE_SHA)"
test -z "$(git -C source status --porcelain)"
(cd source && git ls-files -z | xargs -0 sha256sum) >source.sha256
find . -maxdepth 1 -type f ! -name inputs.sha256 ! -name outputs.sha256 -print0 |
    sort -z | xargs -0 sha256sum >inputs.sha256
find environment -type f -print0 | sort -z | xargs -0 sha256sum >>inputs.sha256
if [ -d reference ]; then find reference -type f -print0 | sort -z | xargs -0 sha256sum >>inputs.sha256; fi
echo 'Inputs and source sealed; do not change the run after submission.'
