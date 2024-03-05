#add pca code to top of this

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))



scaler = MinMaxScaler(feature_range = (0,1))
pca_data = scaler.fit_transform(pca_data)


# input dataset where each row is a training example
X = np.empty((200,2))
for i in range(200):
    X[i] = pca_data[i]
    


# output dataset where each row is a training example
y = np.empty((200,1))
for i in range(200):
    y[i] = mean_numbers[i+1][0]
    


print(X)

y = y.reshape(-1,1)
y = scaler.fit_transform(y)



np.random.seed(1)
syn0 = 2*np.random.random((2,1)) - 1


for iter in np.arange(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    
    l1_error = y - l1
    
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    
print("Output After Training:")
print(l1_error)




predict_data=np.empty((49,2))
for i in range(49):
    predict_data[i] = pca_data[i+200]
    


prediction=nonlin(np.dot(predict_data,syn0))
prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

error = []
error = np.array(error)
error = mean_numbers

err = 0
for i in range(49):
    err += abs(prediction[i] - error[i+201][0])
err = err/49
print(err)


time_slice = np.linspace(200, 249, 49)  
mean_numbers_slice = mean_numbers[200:249, 0]

# Plotting
pyplot.plot(time_slice, mean_numbers_slice)
pyplot.plot(time_slice, prediction)
pyplot.show()
