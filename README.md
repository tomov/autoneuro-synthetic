Agents for generating synthetic data for the automated neuroscientist

Setup
```
conda create -n synthetic python=3.9
conda activate synthetic
pip install -r requirements.txt
```

Generate data
```
python -m experiments.gen_classical
```