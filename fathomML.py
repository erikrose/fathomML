#!/usr/bin/env python

import torch
from torch import no_grad, randn, tensor
from torch.nn import Sequential, Linear, ReLU, MSELoss, BCEWithLogitsLoss


# Training data for NAND.
x = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = tensor([[0], [0], [0], [1]], dtype=torch.float)

# Define a neural network using high-level modules.
model = Sequential(
    Linear(2, 1, bias=True),   # 2 dims (with bias) -> 1 dim. 2 inputs -> 1 output?
)

# sigmoid then binary cross-entropy loss
loss_fn = BCEWithLogitsLoss(size_average=False)

learning_rate = 0.5
for t in range(500):
    y_pred = model(x)                   # Make predictions.
    loss = loss_fn(y_pred, y)           # Compute the loss.
    #print(t, loss.item())

    model.zero_grad()                   # Zero-clear the gradients.
    loss.backward()                     # Compute the gradients.

    with no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad   # Update the parameters using SGD.


# Apply it:
print(model(tensor([[0, 0]], dtype=torch.float)).sigmoid().item())
