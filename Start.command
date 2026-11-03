#!/bin/bash
# Akkordverktøy – dobbeltklikk for å starte (Mac).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

pause_and_exit() { echo; read -r -p "Trykk Enter for å lukke vinduet …" _; exit "${1:-1}"; }

need_install() {
  [ ! -x .venv/bin/python ] || [ ! -f .venv/.installert ] || [ requirements.txt -nt .venv/.installert ]
}

if need_install; then
  echo
  echo "  ┌──────────────────────────────────────────────────────────┐"
  echo "  │  Første oppstart: installerer Akkordverktøy.             │"
  echo "  │  Dette tar noen minutter (laster ned ca. 1–2 GB).        │"
  echo "  │  Neste gang starter appen med en gang.                   │"
  echo "  └──────────────────────────────────────────────────────────┘"
  echo
  PY_OK=""
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,13))' 2>/dev/null; then
    PY_OK=1
  fi
  if ! command -v uv >/dev/null 2>&1 && [ -z "$PY_OK" ]; then
    echo "• Fant ikke uv eller en passende Python (3.10–3.13) – installerer uv …"
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "! Klarte ikke å installere uv."; pause_and_exit 1; }
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "• Lager Python-miljø med uv …"
    uv venv --python 3.12 --allow-existing .venv || pause_and_exit 1
    echo "• Installerer avhengigheter …"
    uv pip install --python .venv/bin/python -r requirements.txt || pause_and_exit 1
  else
    echo "• Lager Python-miljø med $(python3 --version) …"
    python3 -m venv .venv || pause_and_exit 1
    .venv/bin/python -m pip install --upgrade pip
    echo "• Installerer avhengigheter …"
    .venv/bin/python -m pip install -r requirements.txt || pause_and_exit 1
  fi
  touch .venv/.installert
  echo
  echo "• Ferdig installert. Første låt laster i tillegg ned modellene (ca. 2 GB)."
fi

.venv/bin/python app.py || pause_and_exit 1
