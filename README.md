# Medical Concept Recognition in Free Text

I build and query a postgreSQL database, then tune and ensemble Random Forest, Logistic Regression and SVM classifiers to predict hospitalization risk for ED patients at triage.

## Data

Data originates from a [retrospective study](https://www.ncbi.nlm.nih.gov/pubmed/30028888) at the Yale New Haven Health System. It includes all adult ED visits between March 2014 and July 2017 from one academic and two community emergency rooms. The dataset is available on [Kaggle](https://www.kaggle.com/maalona/hospital-triage-and-patient-history-data). 

While a total of 972 variables were extracted per patient visit, I use Random Forest to assess each feature's information gain (based on Gini index) and subset to ~50 fields without a performance impact to the model. 

## Methods

**PostgreSQL database management**
- Database and table creation / manipulation through Python using the psycopg2 library

**Hyperparameter tuning and feature importance assessment**
- Exhaustive search over specified Support Vector Machine parameter values for optimized estimator estimator 
- Random Forest feature information gain assessment to optimize feature sizing for model inclusion.

**Machine learning classification**
- Use of Logistic Regression, Random Forest and Support Vector Machine classifiers to predict hospitalization outcome (admitted as inpatient or discharged from ED) and hospitalization risk. 

**ML model ensembling**
- Use of ensembled, weighted voting model to provide a nuanced hospitalization prediction using multiple model outputs.

## Data manipulation and analysis

```
PostgreSQL
psycopg2
numpy
pandas
sklearn
matplotlib
seaborn
```

### Front end and visualization

```
Tableau
TabPy
```

