#!/usr/bin/env bash
set -uo pipefail

experiments=(
  cifar10-patch4
  cifar10-patch8
  cifar100-patch4
  cifar100-patch8
  fashionmnist-patch4
  fashionmnist-patch8
  tiny-imagenet-patch8
  tiny-imagenet-patch16
)

mkdir -p logs
python_bin="${PYTHON:-python3}"
failures=()

for experiment in "${experiments[@]}"; do
  log_file="logs/log_${experiment//-/_}.txt"
  if [[ "${RERUN_COMPLETED:-0}" != "1" && -f "${log_file}" ]] && grep -q "Training finished." "${log_file}"; then
    echo "Skipping ${experiment}; ${log_file} is already complete"
    continue
  fi
  if [[ -f "${log_file}" ]]; then
    backup_file="${log_file%.txt}_incomplete_$(date +%Y%m%d_%H%M%S).txt"
    echo "Preserving incomplete ${log_file} as ${backup_file}"
    mv "${log_file}" "${backup_file}"
  fi
  echo "Running ${experiment}; logging to ${log_file}"
  if "${python_bin}" train.py "${experiment}" --log-file "${log_file}" "$@"; then
    :
  else
    status=$?
    echo "FAILED ${experiment} with exit code ${status}; continuing"
    failures+=("${experiment}:${status}")
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "Failed experiments: ${failures[*]}"
  exit 1
fi
