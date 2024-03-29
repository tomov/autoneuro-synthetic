Agents and environments for generating synthetic data for the automated neuroscientist.

# Setup
```
conda create -n synthetic python=3.9
conda activate synthetic
pip install -r requirements.txt
```

# Generate data

## Classical conditioning
```
python -m experiments.gen_classical
```
Need to edit file to change environments and/or agents.

# Data

## Classical conditioning

Directory structure: `data/classical/<behavioral phenomenon>/`

Example:
```
data/
  classical/
    overshadowing/  
    recovery/  
    second_order_LI/  
    ...
```

Each has subdirectories:
- For input variables: `input/<variable name>.csv`
- For output variables: `output/<model name>/<variable name>.csv`

Each file corresponds to a different variable. Each row corresponds to a (sequential) data point. 

Concatenating all the files (horizontally) for a given behavioral phenomenon (e.g., `data/classical/overshadowing/`) produces a dataset for the automated neuroscientist.

Example for Rescorla-Wagner model:
```
input/
  states.csv
  rewards.csv
output/
  RW/
    rpes.csv
    values.csv
```

In this case, we're trying to recover the function `y = f(x)`, where:
- `x = [state, reward]`
- `y = [rpe, value]`



## Instrumental conditioning

Directory structure: `data/instrumental/<env>/<agent and params/`

To generate, edit the main in `experiments/gen_instrumental.py` (TODO make configurable). NOTE: will overwrite data for same settings.

Run from root:
```
python -m experiments.gen_instrumental
```