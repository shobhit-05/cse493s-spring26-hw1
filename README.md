# Advanced ML HW 1

Linda Jin, Shobhit Gupta

## Part 0 and 1

The file `model.py` contains an implementation of a transformer model. The file `part_0_1_contract.py` contains some function signatures that would make autograding less painful for the TAs. The notebooks `part0.ipynb` and `part1.ipynb` call inference code `inference.py` and training code `train.py` respectively to visualize training and evaluation of the transformer. All part 0 and 1 artifacts and analysis were included on these two notebooks.

## Part 2

The notebook `part2.ipynb` has code to load pretrained models, the AIME dataset, functions for evaluation and has unoptimized code for inference. Customized vLLM inference code was written as `run_one()`. All part 2 artifacts and analysis were included in this notebook.

## Environment
We ran in a conda enviroment with python==3.11, and all python package dependencies are in `requirements.txt` and can be installed by running:
xs
```bash
pip install -r requirements.txt
```
## Artifacts and Model Checkpoints
All files and model checkpoints are in `cse493s-spring26-hw1/out/submission`. Artifacts and traces are organized under their respective subfolders `part*/`.