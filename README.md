# Elephant–Rider Neural Network

## Overview
This project implements a neural architecture inspired by Jonathan Haidt's Elephant–Rider model.

## Files
- elephant_rider_model.py

## Installation (Colab)
```python
!pip install torch
```

## Run in Colab
```python
from elephant_rider_model import ElephantRiderModel
import torch

model = ElephantRiderModel(input_dim=10)
x = torch.randn(1, 10)
y, alpha = model(x)

print("Output:", y)
print("Alpha:", alpha)
```

## Training Example
Run the script directly:
```bash
python elephant_rider_model.py
```

## Insight
Emotional system dominates, rational system corrects via gating.
