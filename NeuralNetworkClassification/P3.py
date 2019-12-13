#!/usr/bin/env python
# coding: utf-8

# In[693]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ### 1a)

# In[602]:


data = pd.read_csv("irisdata.csv")
data


# In[603]:


data23 = data[data['species'] != 'setosa']


# In[604]:


data23.loc[data23['species'] == 'versicolor', 'species'] = 0
data23.loc[data23['species'] == 'virginica', 'species'] = 1
data23["constant"] = [1]*len(in_vec)


# In[605]:


data23


# In[878]:


petal_vec = data23[['petal_length', 'petal_width']]
species_vec = data23['species']


# In[696]:


plt.scatter(petal_vec.values[:,0], petal_vec.values[:,1], c=species_vec.values) 
plt.colorbar()
plt.xlabel('length')
plt.ylabel('width')
plt.show()


# ### 1b)

# In[160]:


def logistic_function(w, x):
    return 1 / (1+(np.e**(-1 * np.dot(w, x))))


# ### 1c)

# In[516]:


def find_decision_boundary(x1, w1, w2, w3):
    return ((-x1 * w1) - w3) / w2


# In[572]:


def plot_decision_boundary(w, vec1, vec2):
    x1 = np.linspace(3,6.9,100)
    x2 = find_decision_boundary(x1, w[0], w[1], w[2])
    plt.plot(x1, x2)
    plt.scatter(vec1.values[:,0], vec1.values[:,1], c=vec2.values) 
    plt.colorbar()
    plt.show()


# In[746]:


weights = [] #Good weights: [[4.8, 4, -30]]
plot_decision_boundary(weights, petal_vec, species_vec) #0.005, .45, -0.8


# ### 1d)

# In[702]:


test_vec = []
for petal in data23.values:
    hyp = logistic_function(weights, [petal[2], petal[3], petal[5]])
    test_vec.append(hyp)
data23["classification"] = test_vec
data23


# In[706]:


X = list(data23['petal_length'])

Y = list(data23['petal_width'])

Z = list(data23['classification'])

#Reshape to be 10x10 matrices
x = np.reshape(X, (10, 10))
y = np.reshape(Y, (10, 10))
z = np.reshape(Z, (10, 10))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)

ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_zlabel('Petal Class')

plt.show()


# ### 1e)

# In[707]:


temp_df


# In[587]:


temp_df = data23.loc[[57,135,70,133]]
petal_vec = temp_df[['petal_length', 'petal_width']]
species_vec = temp_df['classification']

plot_decision_boundary(weights, petal_vec, species_vec)


# ### 2a)

# In[725]:


def mse(vectors, weights, classes):
    sse = 0
    for petal,p_class in zip(vectors, classes):
        sse += (logistic(weights, petal) - p_class)**2
    return sse/len(classes)


# ### 2b)

# In[770]:


petal_vec_mod = data23[['petal_length', 'petal_width', 'constant']].values
species_vec_mod = data23['species']


# In[771]:


#MSE with good weights
weights = [4.8, 4, -30]
print("Good MSE = ", mse(petal_vec_mod, weights, species_vec_mod))
plot_decision_boundary(weights, petal_vec, species_vec)


# In[772]:


#MSE with bad weights
weights = [1, 2, -3]
print("Bad MSE = ", mse(petal_vec_mod, weights, species_vec_mod))
plot_decision_boundary(weights, petal_vec, species_vec)


# ### 3a)

# In[904]:


train_network(data23, [3, 1, -20], max_iter=1)


# In[899]:


def train_network(data, weights_init, max_iter=2000, mse_threshold=0.05):
    #Trains a neural network using a gradient descent function
    
    #For plotting decision boundary
    petal_vec = data[['petal_length', 'petal_width']]
    species_vec = data['species']
    
    #For storing MSE per iteration
    cur_mse = 0
    mse_vec = []
    
    #For storing the weights to plot progress
    weights = weights_init
    weights_dict = {}
    
    p_input1_vec = data['petal_length']
    p_input2_vec = data['petal_width']
    p_input3_vec = data['constant']
    p_output_vec = data['species']
    petal_vec_mod = data[['petal_length', 'petal_width', 'constant']].values
    
    for i in range(max_iter):
        #Calculate and store MSE
        cur_mse = mse(data[['petal_length', 'petal_width', 'constant']].values, weights, p_output_vec)
        mse_vec.append(cur_mse)
        
        #If mse reaches the given threshold, exit
        if (cur_mse < mse_threshold):
            break
        
        #Increments each weight by computing gradient change for corresponding feature
        weights_dict[str(i)] = [weights[0], weights[1], weights[2]]
        weights[0] += gradient_function(weights[0], weights, p_input1_vec, p_output_vec, petal_vec_mod)
        weights[1] += gradient_function(weights[1], weights, p_input2_vec, p_output_vec, petal_vec_mod)
        weights[2] += gradient_function(weights[2], weights, p_input3_vec, p_output_vec, petal_vec_mod)
    
    #Plot initial decision boundary and MSE vector
    print("Initial Model weights: ", weights_dict['0'])
    plot_decision_boundary(weights_dict['0'], petal_vec, species_vec)
    
    print("Model MSE over iterations:")
    plt.plot(list(range(0,1)), mse_vec[0])
    plt.xlabel('Iterations')
    plt.ylabel('MSE per iteration')
    plt.show()
    
    #Plot decision boundary and MSE vector during middle of learning
    middle_index = int(len(mse_vec)/2)
    print("Middle Model weights: ", weights_dict[str(middle_index)])
    plot_decision_boundary(weights_dict[str(middle_index)], petal_vec, species_vec)
    
    print("Model MSE over iterations:")
    plt.plot(list(range(0,middle_index)), mse_vec[:middle_index])
    plt.xlabel('Iterations')
    plt.ylabel('MSE per iteration')
    plt.show()
        
    #Plot final decision boundary and MSE vector
    print("Model achieved weights: ", weights)
    plot_decision_boundary(weights, petal_vec, species_vec)
    
    print("Model MSE over iterations:")
    plt.plot(list(range(0,len(mse_vec))), mse_vec)
    plt.xlabel('Iterations')
    plt.ylabel('MSE per iteration')
    plt.show()


# In[905]:


def gradient_function(w, weights, p_input_vec, p_output_vec, petal_vec, learning_p=0.05):
    #Accumulator for change in start weight
    weight_cumulator = 0
    
    #Iterates through each petal where
    #  z - feature value of current petal
    #  y - given class of current petal
    #  z - set of features for the current petal
    for x, y, z in zip(p_input_vec, p_output_vec, petal_vec):
        weight_cumulator += learning_p*(y-logistic_function(weights, z)) * logistic_function(weights, z)*(1-logistic_function(weights, z)) * x
    
    return weight_cumulator


# ### 2e)

# In[893]:


weights = [4, 4, -34] #Good weights: [[4.8, 4, -30]]
plot_decision_boundary(weights, petal_vec, species_vec) #0.005, .45, -0.8


# In[886]:


p_input1_vec = data23['petal_length']
p_input2_vec = data23['petal_width']
p_input3_vec = data23['constant']
p_output_vec = data23['species']
petal_vec_mod = data23[['petal_length', 'petal_width', 'constant']].values


# In[894]:


weights[0] += gradient_function(weights[0], weights, p_input1_vec, p_output_vec, petal_vec_mod)
weights[1] += gradient_function(weights[1], weights, p_input2_vec, p_output_vec, petal_vec_mod)
weights[2] += gradient_function(weights[2], weights, p_input3_vec, p_output_vec, petal_vec_mod)


# In[895]:


print("New Weights: ", weights)
plot_decision_boundary(weights, petal_vec, species_vec) #0.005, .45, -0.8


# ### Extra Credit

# In[914]:


def ExtraCredit():
    dataframe = pd.read_csv('./irisdata.csv')
    two_class = dataframe[dataframe['species'] != 'setosa']
    two_class.loc[two_class['species'] == 'virginica', 'species'] = 0 
    two_class.loc[two_class['species'] == 'versicolor', 'species'] = 1
    in_vec = two_class[['petal_length', 'petal_width']]
    out_vec = two_class['species']
    plt.scatter(in_vec.values[:,0], in_vec.values[:,1], c=out_vec.values) 
    plt.colorbar()
    plt.show()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    num_in = 2 # size of input attributes 
    num_out = 1 # size of output
    class Network(nn.Module): 
        def __init__(self):
            super(Network, self).__init__()
            self.fullyconnected1 = nn.Linear(num_in,num_out)

        def forward(self, x):
            x = self.fullyconnected1(x) 
            x = F.sigmoid(x)
            return x
    model = Network()
    criterion = nn.MSELoss() # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100 # number of training iterations 
    num_examples = two_class.shape[0]
    model.train()
    for epoch in range(num_epochs):
        for idx in range(num_examples):
            # for example `idx`, convert data to tensors so that PyTorch can use it.
            attributes = torch.tensor(in_vec.iloc[idx].values, dtype=torch.float)
            label = torch.tensor(out_vec.iloc[idx], dtype=torch.float)
            # reset the optimizer's gradients
            optimizer.zero_grad()
            # send example `idx` through the model
            output = model(attributes)
            # compute gradients based on error
            loss = criterion(output, label)
            # propegate error through network loss.backward()
            # update weights based on propegated error
            optimizer.step() 
        if(epoch % 100 == 0):
            print('Epoch: {} | Loss: {:.6f}'.format(epoch, loss.item()))

    #Test it
    model.eval()
    pred = torch.zeros(out_vec.shape) 
    for idx in range(num_examples):
        attributes = torch.tensor(in_vec.iloc[idx].values, dtype=torch.float)
        label = torch.tensor(out_vec.iloc[idx], dtype=torch.float)
        # save the predicted value
        pred[idx] = model(attributes).round()
    print('Correct classifications: {}/{}'.format(sum(pred == torch.tensor(out_vec.values))))

