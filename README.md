# ML_HW2

## Introduction
This project predicts whether patients diagnosed with HCC will survive a year later.

## Dataset
The dataset contains 49 features from 165 real patients diagnosed with HCC. The features are selected according to the EASL-EORTC Clinical Practice Guidelines, which are the current state-of the art on the management of HCC.   
[DATASET LINK](https://archive.ics.uci.edu/ml/datasets/HCC+Survival)

name | type | abbreviate | range | missing value(%) | Korean name
--- | --- | --- | --- | --- | ---
Gender | nominal | Gender | (1=Male;0=Female) | 0 | 성별
Symptoms | nominal | Symptoms | (1=Yes;0=No) | 10.91 | 증상
Alcohol | nominal | Alcohol | (1=Yes;0=No) | 0 | 음주 여부
Hepatitis B Surface Antigen | nominal | HBsAg | (1=Yes;0=No) | 10.3 | B형간염표면항원
... | ... | ... | ... | ... | ...


## Analysis
For detailed description and analysis on the dataset, see [ML]HW2_Report.pdf   

## How to run
All the charts and results on [ML]HW2_Report.pdf are from hcc_survivor.ipynb.   
To run this:   
- Move to the folder   
- type "jupyter notebook" on terminal   
- Kernel > Restart & Run All   
Note that jupyter notebook or corresponing tool should be installed.   
[![Jupyter Notebook](/Resources/jupyter_notebook.png)]

For those who cannot run ipynb, I also prepare pure python file with resulting scores only.
To run this:
- Move to the folder   
- type "python3  compatibility.py" on terminal   
[![Python file](/Resources/python_result.png)]