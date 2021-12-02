import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Creation(X, Y, n):
    Y=Y.reshape(m,1)
    X=np.append(np.ones([m,1]), (X).reshape(m,1),  axis=1)   
    Y=np.log((Y))
    return X, Y

files=pd.read_csv(r'C:\Users\USUARIO\Desktop\projeto python\casesBrazil.csv',header=None)
X=files[0].values
Y=files[1].values
m=Y.shape[0]
X,Y=Creation(X, Y, 2)
A=X.transpose()
theta=np.dot(np.dot(np.linalg.inv(np.dot(A, X)),A),Y)

plt.scatter(X[:,1],np.exp(Y),c='yellow',marker='x',label='Dados de Treinamento')
plt.plot(X[:,1],(np.exp(theta[0,0]))*np.exp(theta[1,0]*X[:,1]),label='Curva obtida')
plt.ylabel('cases')
plt.xlabel('day')
plt.legend()
plt.title('Curva Obtida por Regressão Não-Linear')
plt.show()
