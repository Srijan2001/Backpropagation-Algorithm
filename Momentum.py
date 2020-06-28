def initialize_velocity(parameters):
    
    L = len(parameters) // 2 		# number of layers in the neural networks
    v = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    
    L = len(parameters) // 2 		# number of layers in the neural networks
    
    for l in range(L):
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)]+(1-beta)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)]+(1-beta)*grads["db" + str(l+1)]
  
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*v["dW" + str(l+1)]		      # update parameters
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*v["db" + str(l+1)]
        
        
    return parameters, v




