#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -v
python run_experiment.py --seeds 20 --seed-offset 1000 --workers 6 --output results
python analyze_results.py
python run_collective_experiment.py --seeds 20 --seed-offset 2000 --workers 6 --output collective_results
make paper

