from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define Stack Model learners and estimator
base_learners = [('knn', KNeighborsClassifier(n_neighbors=5)), ('svc', SVC(kernel='linear'))]
final_estimator = LogisticRegression()

# Stacking model pipeline
stack_model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler()),  # Scale features
    ('stacking', StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5))
])

# Function to train, validate the stacking model
def train_validate(X_train, X_test, X_val, y_train):
    stack_model_pipeline.fit(X_train, y_train)
    validation_predictions = stack_model_pipeline.predict(X_val)
    return validation_predictions

# Function to test the stacking model
def test (X_test):
    test_predictions = stack_model_pipeline.predict(X_test)
    return test_predictions