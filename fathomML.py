#!/usr/bin/env python

from json import load
from sys import argv

from tensorboardX import SummaryWriter
import torch
from torch import no_grad, randn, tensor
from torch.nn import Sequential, Linear, ReLU, MSELoss, BCEWithLogitsLoss


def pages_from_file(filename):
    return load(open(argv[1]))


def training_tensors(filename):
    """Return (inputs, correct outputs) tuple of training tensors."""
    xs = []
    ys = []
    num_targets = 0
    for page in pages_from_file(filename):
        for tag in page['nodes']:
            xs.append(tag['features'])
            ys.append([1 if tag['isTarget'] else 0])  # Tried 0.1 and 0.9 instead. Was much worse.
            if tag['isTarget']:
                num_targets += 1
    return tensor(xs, dtype=torch.float), tensor(ys, dtype=torch.float), num_targets


def training_tensors_per_page(filename):
    raw_vectors = load(open(argv[1]))
    xs = []
    ys = []
    for page in raw_vectors:
        for tag in page['nodes']:
            xs.append(tag['features'])
            ys.append([1 if tag['isTarget'] else 0])  # TODO: try 0.1 and 0.9 instead
    return tensor(xs, dtype=torch.float), tensor(ys, dtype=torch.float)


def learn(x, y, num_targets, run_comment=''):
    # Define a neural network using high-level modules.
    writer = SummaryWriter(comment=run_comment)
    model = Sequential(
        Linear(len(x[0]), len(y[0]), bias=True)  # 9 inputs -> 1 output
    )

    # sigmoid then binary cross-entropy loss
    loss_fn = BCEWithLogitsLoss(size_average=False, pos_weight=tensor([1/(num_targets/len(y))], dtype=torch.float))

    learning_rate = 0.1
    for t in range(500):
        y_pred = model(x)                   # Make predictions.
        loss = loss_fn(y_pred, y)           # Compute the loss.
        writer.add_scalar('loss', loss, t)
        writer.add_scalar('training_accuracy_per_tag', accuracy_per_tag(model, x, y), t)
        #print(t, loss.item())

        model.zero_grad()                   # Zero-clear the gradients.
        loss.backward()                     # Compute the gradients.

        with no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad   # Update the parameters using SGD.
        learning_rate *= .99  # exponential decay: .0007 by 500 iterations

    # Print coeffs:
    print(list(model.named_parameters()))
    #print(model(tensor([[0.9,0.7729160745059742,0.9,0.08,0.9,0.08,0.14833333333333332,0.616949388442898,0.9]], dtype=torch.float)).sigmoid().item())
    #print(model(tensor([[1, 1]], dtype=torch.float)).sigmoid())  # This looks like a probability, as suggested by https://stackoverflow.com/a/43811697. That is, BCE + sigmoid = probability. Confidences for free?
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    return model


def accuracy_per_tag(model, x, y):
    """Return the accuracy 0..1 of the model on a per-tag basis, given input
    and correct output tensors."""
    successes = 0
    for (i, input) in enumerate(x):
        if abs(model(input).sigmoid().item() - y[i].item()) < .001:
            successes += 1
    return successes / len(x)


def accuracy_per_page(model, pages):
    """Return the accuracy 0..1 of the model on a per-page basis."""
    successes = 0
    for page in pages:
        predictions = []
        for tag in page['nodes']:
            prediction = model(tensor(tag['features'],
                                      dtype=torch.float)).sigmoid().item()
            predictions.append({'prediction': prediction,
                                'isTarget': tag['isTarget']})
        predictions.sort(key=lambda x: x['prediction'], reverse=True)
        if predictions[0]['isTarget']:
            print('Success. Confidence:', predictions[0]['prediction'])
            successes += 1
        else:
            print('FAILURE. Confidence:', predictions[0]['prediction'])
            for i, p in enumerate(predictions):
                if p['isTarget']:
                    print('    First success at index', i, p['prediction'])
                    break
    return successes / len(pages)


# Strategy: 1 input neuron for each feature. Train the model on all the features of one tag. Then the next. 1 output neuron, which is "is the price". Or should I have 2 output nodes, 1 for "no" and one for "yes"? Would that help me tell when something has gone wrong?
# Consider: passing a weight= or pos_weight= kwarg to BCEWithLogitsLoss to make the tags that should trigger a 1 output louder. This trades off precision and recall.


def main():
    filename = argv[1]
    if len(argv) > 2:
        run_comment = argv[2]
    else:
        run_comment = ''
    x, y, num_targets = training_tensors(filename)
    model = learn(x, y, num_targets, run_comment=run_comment)
    # [-25.3036,  67.5860,  -0.7264,  36.5506] yields 97.7% per-tag accuracy! Got there with a learning rate of 0.1 and 500 iterations.
    print('Accuracy per tag:', accuracy_per_tag(model, x, y))
    #print('Accuracy per page:', accuracy_per_page(model, pages_from_file(filename)))

if __name__ == '__main__':
    main()
