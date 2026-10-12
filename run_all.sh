#!/bin/bash
# Run every experiment sequentially. Each stage is resumable: a finished run
# leaves its JSON in results/ and is skipped on re-run.
cd "$(dirname "$0")"
export PYTHONPATH=src OMP_NUM_THREADS=2
mkdir -p logs
TXD="--corpus data/corpus_txd22.jsonl --results results/txd22"
M4="--corpus data/corpus.jsonl --results results/m4"
python3 src/run_experiments.py --rq 2        $TXD --epochs 6   # RQ1 + RQ2
python3 src/run_experiments.py --rq grouped  $TXD --epochs 4   # prompt-grouped control
python3 src/run_experiments.py --rq 4        $TXD --epochs 4   # RQ4
python3 src/run_experiments.py --rq 3        $M4  --epochs 4   # RQ3 (M4)
