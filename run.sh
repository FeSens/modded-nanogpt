#!/usr/bin/env bash
set -e

# Run the main training script first
torchrun --standalone --nproc_per_node=8 train_gpt.py "$@"

# Run all other training scripts that match train_gpt*.py except train_gpt.py
for script in train_gpt*.py; do
  if [[ "$script" != "train_gpt.py" && "$script" != "train_gpt_medium.py" ]]; then
    torchrun --standalone --nproc_per_node=8 "$script" "$@"
  fi
done
