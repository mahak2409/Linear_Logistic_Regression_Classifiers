import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import copy
from sklearn.linear_model import LogisticRegression
from numpy import log, dot, e
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)


class LogRegression(object):
    """docstring for LogRegression."""
    
    """beta is to activate L2regularization. Default value as 0(not active)
      multiclassification_type to select between ovr or ovo(Default value as ovr)
      Learning rate can be passed according to the model(Default value as 0.0000005)"""
    def __init__(self,beta=0 , multiclassification_type = 'ovr',learning_rate = 0.0000005):
        super(LogRegression, self).__init__()
        self.beta = beta 
        self.multiclassification_type = multiclassification_type 

        #List to store the coefficients
        self.coefficients = []

        #Dictionary ovo_inde to store which two classes are being compared in one vs one
        self.ovo_index = {}
        self.learning_rate = learning_rate

    #Function gives the score on passed data, index value tells which coefficient in the list to be considered for the calculation
    def sigmoid_calculation(self,X_numpy,index=0):
      z_value = np.dot(X_numpy,self.coefficients[index])
      z_clip = np.clip(z_value,-20,20)
      sigmoid = 1/(1+e**(-z_clip))
      return sigmoid
    
    #Function gives use accuracy 
    def accuracy(self,expected, predicted):
      correct_predictions = expected == predicted
      sum_correct_predictions = correct_predictions.sum()
      accuracy = sum_correct_predictions/expected.shape[0]
      return accuracy
    

    """Function calculates the cost on given data as X &y
    index is used to tell which index coefficient in the list to be considered
    flag is True when we have to append 1 as a column in the data to compute the intercept value, it is False when we already have that"""
    def cost_function(self,X,y,index=0,flag = True):
      if type(X) != np.ndarray:
        X = X.to_numpy()
        
      if type(y) != np.ndarray:
        y = y.to_numpy()
      
      if flag:
        b = np.ones(shape = [X.shape[0],1])
        X = np.concatenate([b,X],axis=1)

      sigmoid = self.sigmoid_calculation(X,index)
      prediction_1 = y * log(sigmoid)
      prediction_0 = (1-y) * log(1-sigmoid)
      cost = np.sum(prediction_1 + prediction_0)+ self.beta*np.dot(self.coefficients[index].T,self.coefficients[index])
      return cost

    

    """Fit function is used to fit models on training data
      test_X and test_y is only expected when we want to calculate accuracy and cost on each iteration on them. 
        Default value for test_X and test_y is None"""
    def fit(self,X,y,test_X = None,test_y = None):
      
      #If unique values in target are 2 then it is a binary classification problem
      if len(np.unique(y))==2:
        #BinaryClass
        return self.fit_binary(X,y,test_X,test_y)
 
      #else the problem is of multiclass type
      else:
        #Getting unique classes from y
        classes = np.unique(y)

        #If multiclassification_type is passed as ovr
        if self.multiclassification_type == 'ovr':
        
          #For loop to get one vs rest 
          for c in classes:

            #Making value as 1 in class_labels when class is c in y and for rest it is 0
            class_labels = (y == c).astype(int)

            #After converting problem into binary classification. fit_binary is called to do the fitting training data
            self.fit_binary(X,class_labels,index = c)
            

        #else ovo will be done for multi class    
        else:
          i=0

          #For loop to do one vs one i.e c1 vs c2
          for c1 in classes:
            for c2 in classes:
              if c1 >= c2 :
                continue
              c1_label = y[y == c1]
              c1_data = X[y==c1]
              c2_label = y[y==c2]
              c2_data = X[y==c2]

              #c1 class is given label as 0
              c1_label_0 = c1_label-c1

              #c2 class is given label as 1
              c2_label_1 = c2_label-(c2-1)

              #c1_label and c2_label are concatenated in labels_contatenate
              labels_concatenate = np.concatenate([c1_label_0,c2_label_1])

              #data of c1 and c2 is concatenated in data_concatenate
              data_concatenate = np.concatenate([c1_data,c2_data])


              #fit_binary is called after converting problem into binary classification again
              self.fit_binary(data_concatenate,labels_concatenate,index = i)

              #ovo_index dictionary is used to keep track at which index of coefficients list two classes are compared
              self.ovo_index[i]=[c1,c2]
              i = i+1
            

    #fit_binary is called when the problem is again a binary classification proble
    def fit_binary(self,X,y,test_X = None,test_y = None,index=0):
      #Lists used to store the loss value iterationwise for training and validation data passed 
      gradient_loss=[]
      gradient_loss_test = []

     #Lists used to store the accuracy value iterationwise for training and validation data passed 
      accuracy = []
      accuracy_test = []

      #list stores the iteration values
      iterations = []

      #Converting X into numpy array if it is not
      if type(X) != np.ndarray:
        X = X.to_numpy()

      #Converting y into numpy array if it is not
      if type(y) != np.ndarray:
        y = y.to_numpy()

      #Appending one in X to do the intercept calculation as well
      b = np.ones(shape = [X.shape[0],1])
      X = np.concatenate([b,X],axis=1)

      #Xavier initialization for initial coefficients
      # scale = 1/398
      # limit = math.sqrt(3*scale)
      # coefficients = np.random.uniform(-limit,limit,size = (X.shape[1]))
      # self.coefficients.append(coefficients)

      #Genrating random coefficients in the beginning of type X.shape[1]
      self.coefficients.append(np.random.rand(X.shape[1]))
      
      #For loop to do 200 iteration for learning the coefficients
      for x in range(200):

        #Sigmoid contains the scores calculated 
        sigmoid = self.sigmoid_calculation(X,index)

        #Update value is being calculates. If beta is active then regularization also gets applied
        update = np.dot(X.T,(y - sigmoid)) + self.beta*2*self.coefficients[index]

        #Updating the existing coefficients with update value and learning rate
        self.coefficients[index] = self.coefficients[index] + ( self.learning_rate * update)

        
        #If test data is present 
        if test_X is not None:
          #Appending loss and accuracy values in the list itertionwise for validation/test data if present
          gradient_loss_test.append(self.cost_function(test_X,test_y,index))
          accuracy_test.append(self.accuracy(test_y,self.predict(test_X)))

          #Appending loss and accuracy values in the list iterationwise for training data
          gradient_loss.append(self.cost_function(X,y,index,False))
          accuracy.append(self.accuracy(y,self.predict(X,False)))
          iterations.append(x+1)


       #Returning iterationwise calculation of loss and accuracy on training and validation data 
      return accuracy,accuracy_test,gradient_loss,gradient_loss_test,iterations

    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""

    """Function is used to give predictions accordingly for binary and multiclass classification
    flag will be true by default if Data(X) is not having ones column required for intercept value calculation , Otherwise it is false"""
    def predict(self,X_test,flag = True):

      #Convert X_test to numpy array if it is not
      if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

      #If flag remains true append ones for the intercept 
      if flag:
        b = np.ones(shape = [X_test.shape[0],1])
        X_test = np.concatenate([b,X_test],axis=1)

      #If length of the coefficient list is 1 then it is a binary classification problem and predictions will be accordingly
      if len(self.coefficients) == 1:
        sigmoid_test = self.sigmoid_calculation(X_test)
        y_predicted = np.round(sigmoid_test)

     #else it is multiclass classification if length is more than 1 for coefficients list   
      else:

        #multiclassification_type is ovr then prediction will be according to that
        if self.multiclassification_type == 'ovr':

          #list will store values for scores for each class vs rest
          scores_classes=[]

          #for loop to calculate scores of sigmoid for each class vs rest
          for c in range(len(self.coefficients)):
            scores_classes.append(self.sigmoid_calculation(X_test,c))
          
          #converting list to numpy array
          scores_classes = np.array(scores_classes)

          #For each instance whichever class gives maximum score is redicted
          y_predicted = np.argmax(scores_classes,axis=0)
        
        #else prediction will be according to ovo
        else:

          #list will sores for each class vs each other class
          scores_classes = []

          #for loop to calculate each class vs each class score list
          for c in range(len(self.coefficients)):
            scores_classes.append(self.sigmoid_calculation(X_test,c))

          #Score list converted to numpy array
          scores_classes = np.array(scores_classes)

          #score class will have class value instead of scores
          predictions_classes = copy(scores_classes)
          for c in range(len(self.ovo_index.keys())):
            scores_classes[c] = np.round(scores_classes[c])
            predictions_classes[c][scores_classes[c]==0] = self.ovo_index[c][0]
            predictions_classes[c][scores_classes[c]==1] = self.ovo_index[c][1]

          predictions_classes = predictions_classes.T
          predictions_classes = predictions_classes.astype(int)

          #list of y_predicted
          y_predicted = []

          #For an instance all class polling value which wins will be given as predicted value
          for c in range(X_test.shape[0]):
            y_predicted.append(np.argmax(np.bincount(predictions_classes[c])))

          #Again list to converted in numpy array
          y_predicted = np.array(y_predicted)

      #return y_predicted value only
      return y_predicted