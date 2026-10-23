#!/usr/bin/env bash
# PTX Downgrader GUI launcher (Linux/macOS)
cd "$(dirname "$0")"
exec python3 -m downgrader --gui
