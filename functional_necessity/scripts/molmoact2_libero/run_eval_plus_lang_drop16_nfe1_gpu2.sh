#!/usr/bin/env bash
# Wrapper: Drop-16 NFE=1 Language on GPU2 (delegates to GPU-parameterized serial + shared flock).
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/run_eval_plus_lang_drop16_nfe1_serial.sh" 2
