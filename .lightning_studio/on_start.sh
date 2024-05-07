#!/bin/bash

# This script runs every time your Studio starts, from your home directory.

# List files under fast_load that need to load quickly on start (e.g. model checkpoints).
#
# ! fast_load
# <your file here>

# Add your startup commands below.
#
# Example: streamlit run my_app.py
# Example: gradio my_app.py
curl https://ollama.ai/install.sh | sh

mkdir -p ~/log
ollama serve > ~/log/ollama.log 2> ~/log/ollama.err &
# preload the model
curl http://localhost:11434/api/generate -d '{"model": "llama3"}'
