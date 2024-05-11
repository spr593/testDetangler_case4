from sklearn.datasets import load_iris
from model import *
from model import *

if __name__=="__main__":
    print("Implement a Stack Model (SM) Learning")
    #print("Model.py will implement the SM functionality in branch b1.")
    #print("Unnittest_model.py will implement the unittest verification of output shapes and f1 scores for the model evaluation in branch b2.")

    #Read Dataset
    X,y = load_iris(return_X_y=True)
    SM = stack_model(X,y)
    training_output = SM.train_validate()
    print(training_output)

    predict_output = SM.predict()
    print(predict_output)

    utest_output = SM.unit_test_model()
    print(utest_output)
