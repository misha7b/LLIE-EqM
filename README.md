# Equilibrium Matching for Low-Light Image Enhancement

Equilibrium Matching (EqM+) applied to paired low-light/normal-light image data.

## Architecture

<p align="center">
  <img src="img/eqmnet.svg" alt="EqM+ Architecture" width="100%" />
</p>

## Usage

```bash
# Train
python -m src.train

# Evaluate
python -m src.eval
```

## Structure

```
src/
  train.py    Training loop
  eval.py     Inference & evaluation
  loss.py
  eqmnet.py   Architecture
datasets/
  paired_dataset.py    Dataset loader
```

## Output Examples

<p align="center">
  <img src="img/8_comparison.png" alt="Comparison example 8" width="48%" />
  <img src="img/9_comparison.png" alt="Comparison example 9" width="48%" />
</p>

<p align="center">
  <img src="img/18_comparison.png" alt="Comparison example 18" width="48%" />
  <img src="img/101_comparison.png" alt="Comparison example 101" width="48%" />
</p>
