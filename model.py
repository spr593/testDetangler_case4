import numpy as np
from sklearn.base import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from unittest_model import TestStackingModel

class stack_model:
    def __init__(self,X,y):
        imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean of the column
        scaler = StandardScaler()  # Scale data to have zero mean and unit variance
        X = imputer.fit_transform(X)
        X = scaler.fit_transform(X)

        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(X, y, test_size=0.3, random_state=33)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_temp, self.y_temp, test_size=0.4, random_state=33)

        # Define Stack Model learners and estimator
        base_learners = [('knn', KNeighborsClassifier(n_neighbors=5)), ('svc', SVC(kernel='linear'))]
        final_estimator = LogisticRegression()

        # Stacking model pipeline
        self.stack_model_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler()),  # Scale features
            ('stacking', StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5))
        ])

    # Function to train, validate the stacking model
    def train_validate(self):
        self.stack_model_pipeline.fit(self.X_train, self.y_train)
        validation_predictions = self.stack_model_pipeline.predict(self.X_val)
        return validation_predictions

    def crossfold_train_validate(self):
        folds = []
        for i in range(self.K):
            folds.append(self.train_validate(self))
        return  np.mean(folds)

    # Function to test the stacking model
    def predict (self):
        test_predictions = self.stack_model_pipeline.predict(self.X_test)
        return test_predictions

    def model_predictions(self):
        validation_predictions = self.train_validate(self.X_train, self.X_test, self.X_val, self.y_train)
        predictions = self.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, predictions)
        validation_accuracy = accuracy_score(self.y_val, validation_predictions)

        test_recall = recall_score(self.y_test, predictions)
        validation_recall = recall_score(self.y_val, validation_predictions)

        test_f1 = f1_score(self.y_test, predictions)
        validation_f1 = f1_score(self.y_val, validation_predictions)
        return validation_predictions, predictions

    def unit_test_model(self):
        unittest = TestStackingModel(self.model_predictions, self.y_val, self.y_test)
        vp , p =  self.model_predictions
        report = unittest.test_model_performance(self, vp,p )
        print(report)
