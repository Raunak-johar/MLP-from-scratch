

import io
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=load_boston();
X=pd.DataFrame(df.data,columns=df.feature_names);
Y=df.target
Y=Y[:,None]



X_train_original,X_test_original,Y_train,Y_test=train_test_split(X,Y,train_size=0.85,random_state=True);

scaler=StandardScaler();
X_train=scaler.fit_transform(X_train_original);



mean=X_train_original.mean(axis=0);
std=X_train_original.std(axis=0);

mean=mean[:,None];
std=std[:,None];

X_train=X_train.transpose();

X_test_original=X_test_original.transpose();

Y_train=Y_train.transpose();
Y_test=Y_test.transpose();

print(X_train.shape)


def standardize(X,mean,std):
  X=(X-mean)/std;
  return X;


def relu(x):
  return np.maximum(0,x);

def sigmoid(x):
  return 1/(1+np.exp(-x));


def deriv_relu(x):
  x=(x>=0);
  return x;


def deriv_sigmoid(x):
  m=sigmoid(x);
  return m*(1-m);


def initialize(layer_nodes):
 L=len(layer_nodes);

 parameters={};

 for l in range(1,L):
  parameters['W'+str(l)]=np.random.randn(layer_nodes[l],layer_nodes[l-1]);
  parameters['b'+str(l)]=np.zeros([layer_nodes[l],1]);

 return parameters;


def forward_propagation(X,parameters):

 W1,W2,b1,b2=parameters['W1'],parameters['W2'],parameters['b1'],parameters['b2'];
  
  L=len(parameters)/2;

  Z1=np.dot(W1,X)+b1; #layer1
  A1=relu(Z1);
  
  Z2=np.dot(W2,A1)+b2;#layer2
  A2=relu(Z2);

  cache={"Z1":Z1,
         "Z2":Z2,
         
         "A0":X,
         "A1":A1,
         "A2":A2}

  return cache;


def compute_cost(Yhat,Y):
  return np.mean( np.power((Yhat-Y)/Y,2));


def backward_propagation(parameters,cache,Y):
  m=Y.shape[1];
  W1,W2,b1,b2=parameters['W1'],parameters['W2'],parameters['b1'],parameters['b2'];
  A0,A1,A2=cache['A0'],cache['A1'],cache['A2'];
  Z1,Z2=cache['Z1'],cache['Z2'];
  L=len(parameters)/2;

  dA2=2*(cache['A2']-Y);
  dZ2=dA2*deriv_relu(Z2);
  dW2=(1/m)*np.dot(dZ2,A1.transpose());
  db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True);
  
  
  dA1=np.dot(W2.transpose(),dZ2);
  dZ1=dA1*deriv_relu(Z1);
  dW1=(1/m)*np.dot(dZ1,A0.transpose());
  db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True);

  gradients={
              'dW2':dW2,
              'dW1':dW1,
             'db1':db1,
             'db2':db2}

  return gradients;


def updation(parameters,gradients,learning_rate=0.0025):
  W1,W2,b1,b2=parameters['W1'],parameters['W2'],parameters['b1'],parameters['b2'];
  dW1,dW2,db1,db2=gradients['dW1'],gradients['dW2'],gradients['db1'],gradients['db2'];
  
  W1=W1-learning_rate*dW1;
  W2=W2-learning_rate*dW2;


  b1=b1-learning_rate*db1;
  b2=b2-learning_rate*db2;


  parameters={"W1":W1,
         "W2":W2,
         
         "b1":b1,
         "b2":b2}
  
  return parameters;



def model_training(X_train):
  parameters=initialize([X_train.shape[0],10,1]);

  for epoch in range(10000):
    
    cache=forward_propagation(X_train,parameters);
    gradients=backward_propagation(parameters,cache,Y_train);
    parameters=updation(parameters,gradients);

    if(epoch%1000==0):
      cost=compute_cost(cache["A2"],Y_train);
      print(cost);
      
  return parameters;


#TRAINING THE MODEL
parameters=model_training(X_train);


