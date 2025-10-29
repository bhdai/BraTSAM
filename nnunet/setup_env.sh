#!/bin/bash

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "Setting up nnU-Net environment for project at: $PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/nnUNet_raw"
mkdir -p "$PROJECT_ROOT/nnUNet_preprocessed"
mkdir -p "$PROJECT_ROOT/nnUNet_results"

# export the environment variables
export nnUNet_raw="$PROJECT_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/nnUNet_results"

echo "nnU-Net environment variables set. You can now run nnU-Net commands."
