Data file downloaded having 32,561 records.
Target is 'Income-Clasification' having two values - <=50k and >50k
This is binary classification problem.

No column has nul values

Features-numeric :
  Age
  FinalWeight
  EducationLevel
  Cap Gain
  Cap Loss
  Hours Per Week

Features-categorical : 
  WorkClass	- 8 unique values including '?'
  Education	- 16 unique values
  Marital Status	- 7 unique values
  Occupation	- 15 unique values including '?'
  Relationship	- 6 unique values
  Race	- 5 unique values
  Sex -	2 unique values
  Native Country -	42 unique values including '?'

Feature Engineering
    1. Scale Age & Hours Per Week between min & max values
    2. Separate rwos with non zero CapGain and CapLoss into new df. There are no rows where both these columns are non zero.
       Then scale this new col between -1 to 1 proportionally
    3. One hot encode all categorical columns

Model Used
    scikit-learn : LogisticRegression

Model Accuracy : 0.795829366

Confusion Matrics :
    0.96456311	0.03543689
    0.7361306	  0.2638694


