#!/usr/bin/env bash
set -uo pipefail

ENV_NAME="nichefate-gpcca"
REPORT_DIR="/home/zhutao/scratch/nichefate/m4d/reports"
LOG_DIR="${REPORT_DIR}/env_logs"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/m4d_01b_setup_standard_gpcca_env_${RUN_ID}.log"
STATUS_REPORT="${REPORT_DIR}/m4d_standard_gpcca_environment_setup_status.md"
FAILURE_REPORT="${REPORT_DIR}/m4d_standard_gpcca_environment_setup_failure.md"
INSTALL_PACKAGES=(
  "python=3.12"
  "pygpcca"
  "cellrank"
  "anndata"
  "scanpy"
  "scipy"
  "numpy"
  "pandas"
  "scikit-learn"
  "matplotlib"
)
IMPORT_MODULES=(
  "pygpcca"
  "cellrank"
  "anndata"
  "scanpy"
  "scipy"
  "numpy"
  "pandas"
  "sklearn"
  "matplotlib"
)

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}"
}

write_failure_report() {
  local reason="$1"
  cat >"${FAILURE_REPORT}" <<EOF
# M4D-01b Standard GPCCA Environment Setup Failure

- time UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- environment: \`${ENV_NAME}\`
- reason: ${reason}
- log file: \`${LOG_FILE}\`
- omicverse modified: \`False\`
- scipy fallback promoted to main algorithm: \`False\`

## Scope Guards
- no full-node GPCCA
- no absorption probability
- no fate probability
- no regulator analysis
- no M5
- no Branched NicheFlow / BranchSBM training
EOF
}

write_status_report() {
  local status="$1"
  rm -f "${FAILURE_REPORT}"
  cat >"${STATUS_REPORT}" <<EOF
# M4D-01b Standard GPCCA Environment Setup Status

- time UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- environment: \`${ENV_NAME}\`
- status: ${status}
- log file: \`${LOG_FILE}\`
- omicverse modified: \`False\`
- scipy fallback promoted to main algorithm: \`False\`

## Algorithm Priority
- primary: pyGPCCA on transition matrices
- secondary: CellRank GPCCA / PrecomputedKernel / custom kernel feasibility
- diagnostic only: \`scipy_pcca_like_diagnostic_fallback\`
EOF
}

env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -qx "${ENV_NAME}"
}

check_imports() {
  local python_code
  python_code='import importlib
mods = ["pygpcca", "cellrank", "anndata", "scanpy", "scipy", "numpy", "pandas", "sklearn", "matplotlib"]
missing = []
for name in mods:
    try:
        module = importlib.import_module(name)
        print("{}={}".format(name, getattr(module, "__version__", "unknown")))
    except Exception as exc:
        missing.append("{}: {}: {}".format(name, type(exc).__name__, exc))
if missing:
    print("MISSING_OR_FAILED_IMPORTS")
    print("\n".join(missing))
    raise SystemExit(1)
'
  conda run --no-capture-output -n "${ENV_NAME}" python -c "${python_code}" >>"${LOG_FILE}" 2>&1
}

run_conda() {
  log "RUN: conda $*"
  conda "$@" >>"${LOG_FILE}" 2>&1
}

log "M4D-01b standard GPCCA environment setup started"
log "Log file: ${LOG_FILE}"

if command -v mamba >/dev/null 2>&1; then
  if mamba --version >>"${LOG_FILE}" 2>&1; then
    log "mamba detected, but setup uses conda for reproducibility"
  else
    log "mamba detected but not usable; setup uses conda"
  fi
else
  log "mamba not found; setup uses conda"
fi

if ! command -v conda >/dev/null 2>&1; then
  log "conda not found"
  write_failure_report "conda executable was not found"
  exit 1
fi

if env_exists; then
  log "Environment ${ENV_NAME} already exists; validating imports without modifying it"
  if check_imports; then
    log "Existing ${ENV_NAME} environment validated successfully"
    write_status_report "existing environment validated"
    exit 0
  fi
  log "Existing ${ENV_NAME} environment failed required import validation"
  write_failure_report "existing ${ENV_NAME} environment is present but required GPCCA packages do not import; not modifying an existing environment automatically"
  exit 1
fi

log "Environment ${ENV_NAME} does not exist; creating isolated Python 3.12 GPCCA environment"
if ! run_conda create -y -n "${ENV_NAME}" -c conda-forge "${INSTALL_PACKAGES[@]}"; then
  log "conda create failed for ${ENV_NAME}"
  write_failure_report "conda create/install failed for ${ENV_NAME}; see log"
  exit 1
fi

log "Created ${ENV_NAME}; validating imports"
if ! check_imports; then
  log "Required imports failed after environment creation"
  write_failure_report "required GPCCA package imports failed after environment creation; see log"
  exit 1
fi

log "M4D-01b standard GPCCA environment setup completed successfully"
write_status_report "created and validated"
