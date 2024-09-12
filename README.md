## Setup
```
pip install git+https://github.com/JoaoCalem/Conformal-Sparsemax
```

## Usage
```
from confpred import ConformalPredictor, SparseScore

cp = ConformalPredictor(SparseScore())
cp.calibrate(cal_true, cal_proba, alpha=0.1)

cp.predict(test_pred) # To get conformal prediction sets
cp.evaluate(test_true, test_proba) # To evaluate coverage and average set size
```

For further usage examples with real datasets and models, see example_usage/