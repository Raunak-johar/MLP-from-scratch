# PREDICTION FUNCTION


def predict(X,parameters):
  X_test=standardize(X,mean,std);
  output=forward_propagation(X_test,parameters);
  return output["A2"];

final=forward_propagation(X_train,parameters);
ycap=final["A2"];
m=compute_cost(ycap,Y_train);
print("Train accuracy",1-m);

Yhat=predict(X_test_original,parameters);
c=compute_cost(Yhat,Y_test);
print("Test Accuracy",1-c);
