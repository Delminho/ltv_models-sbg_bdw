# sBG and BdW models for predicting customer retention
Python code for sBG (shifted beta-geometric) model from ["How to Project Customer Retention"](https://www.brucehardie.com/papers/021/) by Fader and Hardie and BdW (beta-discrete-weibull) model from ["“How to Project Customer Retention” Revisited: The Role of Duration Dependence"](https://brucehardie.com/papers/037/) by Fader and Hardie.

> This code is a modified version of the code originally developed by [jdmaturen](https://github.com/jdmaturen) in the repository [shifted_beta_geometric_py](https://github.com/jdmaturen/shifted_beta_geometric_py).

# Usage
`BDWModel` and `SBGModel` are defined in `models.py`.  
To make a prediction, pass input data and number of periods to project to the `.fit()` method of a model:
```python
data = [0.8, 0.65, 0.53, 0.46]
model = SBGModel()    # or BDWModel()
result = model.fit(data, periods=53)
retention_curve = result['retention_curve']
```
You can pass input data as a single retention curve:
```python
data = [0.8, 0.65, 0.53, 0.46]    # or [1000, 800, 650, 530, 460]
result = model.fit(data, periods=53)
```
Or as separate cohorts:
```python
data = [
  [733, 379, 282, 225],
  [519, 286, 194],
  [557, 292]
]
result = model.fit(data, periods=53)
```
Output is a dictionary with values of parameters, retention curve and the value of loss (log likelihood) function
```python
{'alpha': 3.7906872033733663,
 'beta': 15.160839959123091,
 'retention_curve': [1, 0.7999798554031661, 0.6479878110781834, 0.5307496219788782, 0.43909742004607294, ... ],
 'loss': 1.4399582038997103}
```
