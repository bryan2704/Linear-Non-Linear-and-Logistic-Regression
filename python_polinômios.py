import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Creation(X, Y, n):
    A=X.reshape(m,1)
    Y=Y.reshape(m,1)
    X=np.append(np.ones([m,1]), (X).reshape(m,1),  axis=1)
    for i in range(2,n+1):
        X=np.append(X, A**(i), axis=1)   
    for j in range(1,n+1):
        X[:,j]=(X[:,j]-np.amin(X[:,j]))/(np.amax(X[:,j])-np.amin(X[:,j]))
    Y=(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))    
    return X, Y

def Grad_Descent(X, Y, z, c, n):
    theta=np.zeros((n+1,1))
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

def Prediction(v, a, b, theta, t, h):
    v=(v-a)/(b-a)
    K=[]
    for i in range(0,n+1):
        K.append(v**i)
    K=np.array(K)
    K=K.reshape(1,n+1)
    K=K.dot(theta)[0,0]
    return K*(h-t)+t
    
    
files=pd.read_csv(r'C:\Users\USUARIO\Desktop\projeto python\casesBrazil.csv',header=None)
X=files[0].values
Y=files[1].values
m=Y.shape[0]
a=np.amin(X)
b=np.amax(X)
t=np.amin(Y)
h=np.amax(Y)
n=int(input('Qual seria o grau do polinômio? (Entre 1 e 10) '))
r=float(input('Qual seria a taxa de aprendizado? (Para n>=7, recomendamos <=0.88) '))
o=int(input('Qual seria o número de iterações? '))
l=int(input('Tecle 1 para ver a curva obtida pelo método ou 0 para ver a curva de custos: '))
p=int(input('Tecle um dia para o qual gostaria de fazer uma previsão, caso o contrário, tecle 0: '))
X,Y=Creation(X, Y, n)
theta,J=Grad_Descent(X, Y, o, r, n)


if p!=0:
   print('\n')
   print('A previsão para o dia',p, 'é de:',int(Prediction(p, a, b, theta, t, h)), 'casos')

if l==1:
    plt.scatter(X[:,1]*134,Y*1.75*10**6,c='yellow',marker='x',label='Dados de Treinamento')
    plt.plot(X[:,1]*134,np.dot(X,theta)*1.75*10**6,label='Curva obtida')
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