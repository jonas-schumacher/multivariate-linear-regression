# Multivariate Linear Regression using second-order optimizer L-BFGS
A comparison between the analytic approach and gradient descent using first- and second-order optimizers in PyTorch.

## Documentation
The full documentation & motivation for this project can be found in my Medium story titled: 
[Multivariate Linear Regression using second-order optimizer L-BFGS](https://medium.com/@jonas.schumacher/multivariate-linear-regression-using-second-order-optimizer-l-bfgs-95ca229cd369).

## How to get started
- Clone this repository
- Create a virtual environment of your choice using Python 3.12
- A note on the Python version: I tested my code with Python 3.12, but it should run equally well with any version between 3.8 & 3.12 (3.13 was not yet supported by PyTorch at the moment of writing)
- Install required packages: `pip install -r requirements.txt`
- Run the code: `python multivariate_regression.py`
- If you want to run the jupyter notebook, you need to additionally install: `pip install notebook`

## Experiments documented in the Medium story referenced above
Experiment #1 (baseline with perfect predictions)
```
NUM_SAMPLES = 5
ADD_UNEXPLAINABLE_NOISE_TO_TARGETS = False
USE_ILL_CONDITIONED_FEATURE_MATRIX = False
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE_SGD = 0.5
LEARNING_RATE_LBFGS = 0.5
```

Experiment #2 (add noise to target t2)
```
NUM_SAMPLES = 5
ADD_UNEXPLAINABLE_NOISE_TO_TARGETS = True
USE_ILL_CONDITIONED_FEATURE_MATRIX = False
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE_SGD = 0.5
LEARNING_RATE_LBFGS = 0.5
```

Experiment #3 (increased sample size)
```
NUM_SAMPLES = 100
ADD_UNEXPLAINABLE_NOISE_TO_TARGETS = True
USE_ILL_CONDITIONED_FEATURE_MATRIX = False
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE_SGD = 0.5
LEARNING_RATE_LBFGS = 0.5
```

Experiment #4 (use ill-conditioned feature matrix)
```
NUM_SAMPLES = 100
ADD_UNEXPLAINABLE_NOISE_TO_TARGETS = True
USE_ILL_CONDITIONED_FEATURE_MATRIX = True
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE_SGD = 0.5
LEARNING_RATE_LBFGS = 0.5
```

Experiment #5 (reduced learning rate for SGD)
```
NUM_SAMPLES = 100
ADD_UNEXPLAINABLE_NOISE_TO_TARGETS = True
USE_ILL_CONDITIONED_FEATURE_MATRIX = True
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE_SGD = 0.2
LEARNING_RATE_LBFGS = 0.5
```
