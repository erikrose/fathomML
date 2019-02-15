#!/usr/bin/env python

import torch
from torch import no_grad, randn, tensor
from torch.nn import Sequential, Linear, ReLU, MSELoss, BCEWithLogitsLoss


# Training data for NAND.
x = tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.float)
y = tensor([[0], [0], [0], [1], [1], [1], [0]], dtype=torch.float)

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
print(model(tensor([[1, 1]], dtype=torch.float)).sigmoid())  # This looks like a probability, as suggested by https://stackoverflow.com/a/43811697. That is, BCE + sigmoid = probability. Confidences for free?



# Strategy: 1 input neuron for each feature. Train the model on all the features of one tag. Then the next. 1 output neuron, which is "is the price". Or should I have 2 output nodes, 1 for "no" and one for "yes"? Would that help me tell when something has gone wrong?
# Consider: passing a weight= or pos_weight= kwarg to BCEWithLogitsLoss to make the tags that should trigger a 1 output louder. This trades off precision and recall.