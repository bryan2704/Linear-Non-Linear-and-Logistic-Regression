import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Load():
    X = data_2.values
    m = X.shape[0]
    n = X.shape[1]
    nbr_classes = 10
    X = np.append(np.ones([m,1]),X,axis=1)
    y = data_3.replace(10,0).values
    aux = []
    for i in range(m):
        aux.append(np.array([1 if y[i] == j else 0 for j in range(nbr_classes)]))
    Y = np.array(aux).reshape(-1, nbr_classes)
    return X, Y, m, n, nbr_classes

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunctionLogRegReg(X, Y, Theta, beta):
    J = costFunctionLogReg(X, Y, Theta) + np.diag((beta/(2*m))*(Theta[1:].T @ Theta[1:]))
    return J

def costFunctionLogReg(X, Y, Theta):
    m = len(Y)
    h = sigmoid(X.dot(Theta))
    J = (-1/m)*np.diag(Y.T @ np.log(h) + (1-Y).T @ np.log(1-h))
    return J

def gradientDescentLogRegReg(X, Y, Theta, alpha, beta, nbr_iter):
    J_history = []
    m = len(Y)
    Theta_aux = Theta.copy()
    Theta_aux[0] = 0
    for i in range(nbr_iter):
        h = sigmoid(X.dot(Theta))
        Theta_aux = Theta.copy()
        Theta_aux[0] = 0
        Theta = Theta - (alpha/m)*(X.T.dot(h-Y) + beta*Theta_aux)
        J_history.append(costFunctionLogRegReg(X, Y, Theta, beta))
    return Theta, J_history

def predict(X, theta):
    m = X.shape[0]
    aux = sigmoid(X.dot(theta))
    pred = []
    for row in aux:
        pred.append(np.argmax(row))
    
    pred = np.array(pred).reshape((m, 1)) 
    return pred

def evaluatePrediction(Y, pred):
    m = len(Y)
    Y_labels = []
    for row in Y:
        Y_labels.append(np.argmax(row))
        
    Y_labels = np.array(Y_labels).reshape((m, 1))
    ratio = (Y_labels == pred).sum()/m
    return ratio

def missFeedback(X, Y, pred):
    m = len(Y)
    y_labels = []
    for row in Y:
        y_labels.append(np.argmax(row))
    y_labels = np.array(y_labels).reshape((m, 1))
    for i in range(m):
        if not (y_labels == pred)[i]:
            print(f'Label: \t {y_labels[i]}')
            print(f'Predição: {pred[i]}')
            pixels = X[i, 1:].reshape((20, 20))
            plt.imshow(pixels, cmap='gray')
            plt.show()
            print('----------------------------------')        


data_2 = pd.read_csv(r'C:\Users\USUARIO\Desktop\projeto python\imageMNIST.csv', sep=',', decimal=",")
data_3 = pd.read_csv(r'C:\Users\USUARIO\Desktop\projeto python\labelMNIST.csv')
X,Y,m,n,nbr_classes=Load()
Theta = np.zeros([n+1,nbr_classes])
nbr_iter = 3000
alpha = 3
beta = 3
new_Theta, J_history = gradientDescentLogRegReg(X, Y, Theta, alpha, beta, nbr_iter)
pred=predict(X,new_Theta)
percentage=evaluatePrediction(Y,pred)
print(f'Taxa de acerto = {100*percentage:.2f}%')
a=int(input("Caso queira ver os números que o modelo identificou erroneamente, tecle 0: "))
if a==0:
    missFeedback(X, Y, pred)