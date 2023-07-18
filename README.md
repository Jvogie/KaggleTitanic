# Titanic: Machine Learning from Disaster

This is a machine learning project where I predict the survival of passengers on the Titanic. The dataset is from the Kaggle competition, Titanic: Machine Learning from Disaster.
## Project Overview

    Data Preprocessing: The project starts with data cleaning and preprocessing where I handle missing values 
    and convert categorical variables into numerical ones. I also drop some features which do not contribute 
    much to the predictive model.

    Model Building: I use the XGBoost Classifier for the prediction task. It is a powerful model 
    known for its performance and speed.

    Hyperparameter Tuning: I use RandomizedSearchCV to tune the hyperparameters of the XGBoost model 
    to find the optimal parameters for this dataset.

## Requirements

The main libraries used in this project are:

    Pandas
    NumPy
    Scikit-Learn
    XGBoost
    

## Files

    train.csv: the training set
    test.csv: the test set
    gender_submission.csv: a set of predictions that assume all and only female passengers survive, as an 
    example of what a submission file should look like
    submission.csv: my submission file

## Results

The final model achieved an accuracy of 84% on the validation set. I then used the model to make a survival prediction on the test set which gave me an accuracy of 77%


## Left Over Thoughts

  This was a simple introdution to kaggle competitions so I didn't go crazy testing models, hyperparameters, or fiddling with the data. It was also my first time using XGBoost. 
  However, there are some things that I would like to do if I come back to this such as feature engineering and trying different models so that I can get a better accuracy on the submission file.
