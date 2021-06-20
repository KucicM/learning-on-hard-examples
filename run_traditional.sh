#!/bin/bash

source venv/bin/activate

seed=42
shuffle_portions=(0.0 0.1 0.2 0.3 0.4)

for shuffle in ${shuffle_portions[*]}
do
    python main.py --method="traditional" --seed=$seed --shuffle-proportion=$shuffle
done

