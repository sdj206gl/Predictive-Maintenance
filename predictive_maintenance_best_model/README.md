---
library_name: scikit-learn
tags:
- predictive-maintenance
- classification
- mlops
- gradientboosting
---

# Predictive Maintenance Model

## Model Description
This is a GradientBoosting model trained to predict engine failure based on sensor readings.

## Model Performance
- Algorithm: GradientBoosting
- Task: Binary Classification
- Dataset: Engine Condition Sensor Data
- Accuracy: 0.6678

## Usage
```python
import joblib
model = joblib.load('model.pkl')
predictions = model.predict(X)
```

## Features
Engine RPM, Lub Oil Pressure, Fuel Pressure, Coolant Pressure, Lub Oil Temp, Coolant Temp
