"""
Created by Sri V.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import frequency

# [Training data]
datax = np.linspace(0, 2 * np.pi, 100) # generates 100 evenly spaced points from 0 to 2pi
datay = np.sin(datax) # function

# Tensoring conversion
dataxTensor = torch.tensor(datax, dtype=torch.float32).unsqueeze(1)
datayTensor = torch.tensor(datay, dtype=torch.float32).unsqueeze(1)

def pe1D(p,L):
    p = np.asarray(p)
    # Generates frequencies from -(L-1) to (L-1)
    # Low frequency components catch slow variations
    # High frequency components catch finer details
    frequency = np.linspace(-(L-1)*np.pi, (L-1)*np.pi, num=L)
    # frequency = 2.0**torch.arange(0, L, dtype=torch.float32) * np.pi
    # Frequency Response with scalar weights to prevent extreme values
    # Can possibly modify or generalize
    encoded = np.concatenate([np.sin(0.1*np.pi * frequency * p)/10, np.cos(0.1*np.pi *
                                                                            frequency *
                              p)/10],
                             axis=-1)
    # "Uniformizer" aka the unit vector
    encoded /= np.linalg.norm(encoded, ord=2, axis=-1, keepdims=True)
    return encoded

# [Apply PE into Tensor]
L = 6 # 6 frequency components
encodedX = np.array([pe1D(p, L) for p in datax]) # final vector will have 2L dimensions
encodedXTensor = torch.tensor(encodedX, dtype=torch.float32)

class MLP(nn.Module):
    """
    * Creating a neural network
    * If dim = 1, then it is raw
    * If dim = 2L, then it will be using PE.
    *
    * Goes from dim -> 32 -> 32 -> 1
    * fci is a fully connected layer
    """
    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.ReLU6()
        self.outputActivation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = self.outputActivation(x)
        return x

def train(datax, datay, dim, epochs=1000, lr=0.01):
    model = MLP(dim)
    # Weight decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # MSELoss = Mean Square Error since we are doing regression
    lossFn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        yPred = model(datax)
        loss = lossFn(yPred, datay)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:  # Print loss every 100 epochs
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return model

raw = train(dataxTensor, datayTensor, dim=1, lr=0.01) # dim 1 Raw
encoded = train(encodedXTensor, datayTensor, dim=2*L) # dim 2L -> PE

testX = np.linspace(0, 4 * np.pi, 200)
testY = np.sin(testX)
testXTensor = torch.tensor(testX, dtype=torch.float32).unsqueeze(1)
encodedTestX = np.array([pe1D(p,L) for p in testX])
encodedTestXTensor = torch.tensor(encodedTestX, dtype=torch.float32)

yPredRaw = raw(testXTensor).detach().numpy()
yPredEncoded = encoded(encodedTestXTensor).detach().numpy()

# [Plot]
plt.figure(figsize=(10,5))
plt.plot(testX, testY, label='True Function (sin(x))', linestyle="dashed")
plt.plot(testX, yPredRaw, label="Without PE", alpha=0.7)
plt.plot(testX, yPredEncoded, label="With PE", alpha=0.7)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Function Regression")
plt.show()