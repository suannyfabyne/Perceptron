import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def weight_init(num_inputs): 
    """
    Função que inicializa os pesos e bias aleatoriamente utilizando numpy
    Parâmetro: num_inputs - quantidade de entradas X
    Retorna: w,b - pesos e bias da rede inicializados
    """
    ### Insira seu código aqui (~2 linhas)
    w = np.random.random_sample(num_inputs) - 0.5
    return w

def activation_func(func_type, z):

	if func_type == 'degrau':
		if (z>0): 
			return 1 
		else: 
			return 0

	elif func_type == 'sigmoid':
		return 1 / (1 + (np.exp(-z))) 

	elif func_type == 'relu':
		return max(0, z)

	elif func_type == 'tanh':
		return (2 / (1 + np.exp(-2*z))) - 1
	else: 
		return 'Is not a function'

# z = np.arange(-5., 5., 0.2)
# def visualizeActivationFunc(z):
#    z = np.arange(-5., 5., 0.2)
#    func = []
#    for i in range(len(z)):
#        func.append(activation_func('degrau', z[i]))
#
#    plt.plot(z,func)
#    plt.xlabel('Entrada')
#    plt.ylabel('Valores de Saída')
#    plt.show()
#visualizeActivationFunc(z)

def forward(w,b,X):
	z = np.dot(w,X) + b
	out = activation_func('relu', z)
	return out


def predict(out):
	threshold = 0.5
	if out <= threshold:
		return 0
	else:
		return 1

def perceptron(x,y, num_interaction, learning_rate):
	w = weight_init(2);
	b = 1;
	print("Peso", w)

	for it in range(num_interaction):
		print ("ITERAÇÃO", it+1)
		for j in range(len(x)):
			y_pred = forward(w, b, x[j])
			error = y[j] - y_pred

			w = w + (np.float64(x[j])*error*learning_rate)

	print('Saída obtida:', y_pred)
	print('Pesos obtidos:', w)
    #Métricas de Avaliação
	y_pred = predict(y_pred)
	print('Predição', y_pred)
	#print('Matriz de Confusão:')
	#print(confusion_matrix(y, y_pred))
	#print('F1 Score:')
	#print(classification_report(y, y_pred))

		  #X1    X2      
matrix =[[0.08,	0.72],
		 [0.10,	1.00],
		 [0.26,	0.58],
		 [0.35,	0.95],
		 [0.45,	0.15],
		 [0.60,	0.30],
		 [0.70,	0.65],
		 [0.92,	0.45]]

y = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]




perceptron(matrix, y, 10, 0.8)
