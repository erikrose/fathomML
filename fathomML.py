#!/usr/bin/env python

from json import load
from sys import argv

from tensorboardX import SummaryWriter
import torch
from torch import no_grad
from torch.nn import Sequential, Linear, ReLU, MSELoss, BCEWithLogitsLoss, L1Loss


def tensor(some_list):
    """Cast a list to a tensor of the proper type for our problem."""
    return torch.tensor(some_list, dtype=torch.float)


def pages_from_file(filename):
    return load(open(filename, 'r'))


def tensors_from(filename):
    """Return (inputs, correct outputs, number of tags that are recognition targets)
    tuple of training tensors."""
    xs = []
    ys = []
    num_targets = 0
    for page in pages_from_file(filename):
        for tag in page['nodes']:
            xs.append(tag['features'])
            ys.append([1 if tag['isTarget'] else 0])  # Tried 0.1 and 0.9 instead. Was much worse.
            if tag['isTarget']:
                num_targets += 1
    return tensor(xs), tensor(ys), num_targets


def learn(x, y, validation_ins, validation_outs, run_comment=''):
    # Define a neural network using high-level modules.
    writer = SummaryWriter(comment=run_comment)
    model = Sequential(
        Linear(len(x[0]), len(y[0]), bias=True)  # 9 inputs -> 1 output
    )

    loss_fn = BCEWithLogitsLoss(reduction='sum')  # reduction=mean converges slower.

    learning_rate = .1
    for t in range(1000):
        y_pred = model(x)                   # Make predictions.
        loss = loss_fn(y_pred, y)           # Compute the loss.
        writer.add_scalar('loss', loss, t)
        writer.add_scalar('validation_loss', loss_fn(model(validation_ins), validation_outs), t)
        writer.add_scalar('training_accuracy_per_tag', accuracy_per_tag(model, x, y), t)
        writer.add_scalar('avg_abs_offness_per_tag', offness_per_tag(model, x, y), t)
        # See if values are getting super small or large and floating point
        # precision limits are taking over and making the loss function grow:
        writer.add_scalar('coeff_abs_sum', list(model.parameters())[0].abs().sum().item(), t)

        model.zero_grad()                   # Zero-clear the gradients.
        loss.backward()                     # Compute the gradients.

        with no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad   # Update the parameters using SGD.
        #learning_rate *= .99

    # Print coeffs:
    print(list(model.named_parameters()))
    # Horizontal axis is what confidence. Vertical is how many samples were that confidence.
    writer.add_histogram('confidence', confidences(model, x), t)
    writer.close()
    return model


def offness_per_tag(model, x, y):
    """Return the average absolute offness of the prediction."""
    offness = 0
    for (i, input) in enumerate(x):
        offness += abs(model(input).sigmoid().item() - y[i].item())
    return offness / len(x)


def accuracy_per_tag(model, x, y):
    """Return the accuracy 0..1 of the model on a per-tag basis, given input
    and correct output tensors."""
    successes = 0
    for (i, input) in enumerate(x):
        if abs(model(input).sigmoid().item() - y[i].item()) < .5:  # TODO: Change to .5 to not demand such certainty.
            successes += 1
    return successes / len(x)


def confidences(model, x):
    return model(x).sigmoid()


def accuracy_per_page(model, pages):
    """Return the accuracy 0..1 of the model on a per-page basis."""
    successes = 0
    for page in pages:
        predictions = []
        for tag in page['nodes']:
            prediction = model(tensor(tag['features'])).sigmoid().item()
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
    training_file = argv[1]
    validation_file = argv[2]
    if len(argv) > 3:
        run_comment = argv[3]
    else:
        run_comment = ''
    x, y, _ = tensors_from(training_file)
    validation_ins, validation_outs, _ = tensors_from(validation_file)
    model = learn(x, y, validation_ins, validation_outs, run_comment=run_comment)
    # [-25.3036,  67.5860,  -0.7264,  36.5506] yields 97.7% per-tag accuracy! Got there with a learning rate of 0.1 and 500 iterations.
    print('Training accuracy per tag:', accuracy_per_tag(model, x, y))
    print('Validation accuracy per tag:', accuracy_per_tag(model, validation_ins, validation_outs))
    #print('Accuracy per page:', accuracy_per_page(model, pages_from_file(training_file)))


if __name__ == '__main__':
    main()
