## Dataset

### Abstract

### Exploratory Data Analysis

A couples of observations are made

- `?` is used for `na` value
- some classes are imbalanced


### Missing Values

| name              |   %    |
|-------------------|--------|
| race              | 2.234  |
| weight            | 96.858 |
| payer_code        | 39.557 |
| medical_specialty | 49.082 |
| diag_1            | 0.021  |
| diag_2            | 0.352  |
| diag_3            | 1.398  |

Looking at the missing values, I will:

 - drop columns: weight, payer_code. medical_speciality
 - drop na rows for diag_1, diag , diag_3, race 
 
 
 ## Variation 
 
 
 