import os
import re
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn as nn
from torch import autograd as autograd
from torch.nn import functional as F
from torch.autograd import Variable
from torchtext import data


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--predict", type=str, default=None)
parser.add_argument("-m", "--batch_size", type=int, default=64)
args = parser.parse_args()

if args.predict is None:
    raise ValueError("No predict text.")


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()

    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)

    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)

    return label_feild.vocab.itos[predicted.data[0] + 1]


label = predict(predict, model, text_field, label_field)
print('\n[Text]  {}\n[Label] {}\n'.format(predict, label))