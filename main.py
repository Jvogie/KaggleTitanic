import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Dropping ticket, name, and cabin columns because they do not provide much predictive power
train_data = train_data.drop(['Name', 'Cabin', 'Ticket'], axis=1)
test_data = test_data.drop(['Name', 'Cabin','Ticket'], axis=1)

# Select the columns to one hot encode
columns_to_encode = ['Sex', 'Embarked']

for column in columns_to_encode:
    # One hot encode the training data
    one_hot = pd.get_dummies(train_data[column], dtype=int)
    # Drop the original column from the training data
    train_data = train_data.drop(column, axis=1)
    # Add the one hot encoded data to the training data
    train_data = train_data.join(one_hot)

    # Repeat for the test data
    one_hot = pd.get_dummies(test_data[column],dtype=int)
    test_data = test_data.drop(column, axis=1)
    test_data = test_data.join(one_hot)

# Print the first 5 rows of the data
# print(train_data.head())
# print(test_data.head())

# Assign our target variable as "Survived" and rest as features
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)


# Model used for finding roughly where best parameters are

# Specify parameter distributions
# param_dist = {'n_estimators': range(100, 1000, 100),
#               'learning_rate': np.linspace(0.01, 0.6, 15),
#               'max_depth': range(3,10),
#               'colsample_bytree': np.linspace(0.5, 1, 6),
#               'subsample': np.linspace(0.5, 1, 6)}
#
# # Create a XGBoost model
# model = xgb.XGBClassifier(use_label_encoder=False, random_state=42)
#
# # Create the RandomizedSearchCV object
# rs = RandomizedSearchCV(estimator=model,
#                         param_distributions=param_dist,
#                         scoring='accuracy',
#                         n_iter=50,
#                         cv=5,
#                         verbose=1,
#                         n_jobs=-1,
#                         random_state=42)
#
# # Fit the RandomizedSearchCV object to the data
# rs.fit(X, y)
#
# # Get the best parameters
# best_params = rs.best_params_
#
# print("Best parameters: ", best_params)


# Best parameters:  {'subsample': 0.5, 'n_estimators': 100, 'max_depth': 4,
# 'learning_rate': 0.01, 'colsample_bytree': 0.7}


#actual model used for prediction

model=xgb.XGBClassifier(use_label_encoder=False,max_depth=6, learning_rate=0.03, n_estimators=200, subsample=0.3, colsample_bytree=0.8)

model.fit(X_train, y_train)

y_val_pred=model.predict(X_val)

accuracy=accuracy_score(y_val,y_val_pred)

print("Validation Accuracy: ", accuracy)


# Make predictions on the test data
test_predictions = model.predict(test_data)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

# Write the DataFrame to a csv file
submission.to_csv('submission.csv', index=False)

