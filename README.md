# Data-Minimization-Auditor 
An auditing tool for model-instability based data minimization that is introduced in "<em>Auditing Black-Box Prediction Models for Data Minimization Compliance</em>" (Bashir Rastegarpanah, Krishna P. Gummadi, Mark Crovella; NeurIPS2021).

The implementation is in Python 3 and the main methods are in ('DM.py'). Given any prediction model that supports the scikit-learn interface style, this tool can be used to perform two types of probabilistic audits at a given confidence level: (i) measuring the greatest level of data minimization that can be guaranteed given a fixed budget of system queries, and (ii) verifying whether data minimization is satisfied at a given level. In the second case, the auditor keeps querying the prediction system until a decision can be made.

A Jupyter notebook that demonstrates how to reproduce the results presented in the experiments section (Sec 6) of the paper is also added to this repository.  In order to use this package one needs to initiate an instance of the 'Auditor()' class whose parameters are a prediction model that provides the same interface as scikit-learn predictors, and some audit data as a pandas DataFrame.
