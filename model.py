from sklearn.base import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
def predict (X_test):
    test_predictions = stack_model_pipeline.predict(X_test)
    return test_predictions

def preprocessing(X, y):
    imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean of the column
    scaler = StandardScaler()  # Scale data to have zero mean and unit variance
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=33)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.4, random_state=33)

def model_predictions(X_train, X_test, X_val, y_train, y_test, y_val):
    validation_predictions = train_validate(X_train, X_test, X_val, y_train)
    predictions = predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    validation_accuracy = accuracy_score(y_val, validation_predictions)