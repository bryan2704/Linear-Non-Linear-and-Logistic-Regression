import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Creation(X, Y, n):
    Y=Y.reshape(m,1)
    X=np.append(np.ones([m,1]), (X).reshape(m,1),  axis=1)
    for j in range(1,n):
        X[:,j]=(X[:,j]-np.amin(X[:,j]))/(np.amax(X[:,j])-np.amin(X[:,j]))    
    Y=(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y)) 
    Y=np.log((Y+0.001))
    return X, Y

def Grad_Descent(X, Y, z, c, n):
    theta=np.zeros((n,1))
    J=[]
    for i in range(z+1):
        b=np.dot(X.transpose(),(X.dot(theta)-Y))
        theta=theta-(c/m)*b
        J.append(Compute_Cost(X, Y, theta))
    return theta, J

def Compute_Cost(X, Y, theta):
 
  h = X.dot(theta)
  J = (1/(2*m))*np.sum((h-Y)**2)
  return J
    
files=pd.read_csv(r'C:\Users\USUARIO\Desktop\projeto python\casesBrazil.csv',header=None)
X=files[0].values
Y=files[1].values
m=Y.shape[0]
r=float(input('Qual seria a taxa de aprendizado? '))
o=int(input('Qual seria o número de iterações? '))
l=int(input('Tecle 1 para ver a curva obtida pelo método ou 0 para ver a curva de custos: '))
X,Y=Creation(X, Y, 2)
theta,J=Grad_Descent(X, Y, o, r, 2)

if l==1:
    plt.scatter(X[:,1]*134,(np.exp(Y)*(1.75*10**6)),c='yellow',marker='x',label='Dados de Treinamento')
    plt.plot(X[:,1]*134,(((np.exp(theta[0,0]))*np.exp(theta[1,0]*X[:,1]))*(1.75*10**6)),label='Curva obtida')
    plt.ylabel('cases')
    plt.xlabel('day')
    plt.legend()
    plt.title('Curva Obtida por Regressão Não-Linear')
    plt.show()
if l==0:
    plt.plot(range(len(J)),J)
    plt.xlabel('Número de iterações')
    plt.ylabel('Custo')
    plt.show()