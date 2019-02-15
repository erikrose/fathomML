#!/usr/bin/env python

from json import load
from sys import argv

import torch
from torch import no_grad, randn, tensor
from torch.nn import Sequential, Linear, ReLU, MSELoss, BCEWithLogitsLoss

def training_tensors(filename):
    """Return (inputs, correct outputs) tuple of training tensors."""
    raw_vectors = load(open(argv[1]))
    xs = []
    ys = []
    for page in raw_vectors:
        for tag in page['nodes']:
            xs.append(tag['features'])
            ys.append([1 if tag['isTarget'] else 0])  # TODO: try 0.1 and 0.9 instead
    return tensor(xs, dtype=torch.float), tensor(ys, dtype=torch.float)


def learn(x, y):
    # Define a neural network using high-level modules.
    model = Sequential(
        Linear(len(x[0]), len(y[0]), bias=True)  # 9 inputs -> 1 output
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

    # Print coeffs:
    print(list(model.named_parameters()))

    # Apply it:
    successes = 0
    for (i, input) in enumerate(x):
        if abs(model(input).sigmoid().item() - y[i].item()) < .001:
            successes += 1
    print('Accuracy:', successes / len(x))
    #print(model(tensor([[0.9,0.7729160745059742,0.9,0.08,0.9,0.08,0.14833333333333332,0.616949388442898,0.9]], dtype=torch.float)).sigmoid().item())
    #print(model(tensor([[1, 1]], dtype=torch.float)).sigmoid())  # This looks like a probability, as suggested by https://stackoverflow.com/a/43811697. That is, BCE + sigmoid = probability. Confidences for free?


# Strategy: 1 input neuron for each feature. Train the model on all the features of one tag. Then the next. 1 output neuron, which is "is the price". Or should I have 2 output nodes, 1 for "no" and one for "yes"? Would that help me tell when something has gone wrong?
# Consider: passing a weight= or pos_weight= kwarg to BCEWithLogitsLoss to make the tags that should trigger a 1 output louder. This trades off precision and recall.


def main():
    x, y = training_tensors(argv[1])
    learn(x, y)


if __name__ == '__main__':
    main()