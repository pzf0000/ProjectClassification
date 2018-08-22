import os
import re
import numpy as np
import pandas as pd
import argparse
import torch
from torchtext import data
from torch import autograd as autograd
from model import TextCNN

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0.5)
parser.add_argument("-ed", "--embed_dim", type=int, default=128)
parser.add_argument("-ks", "--kernel_sizes", type=str, default="[1, 2, 3, 3, 2, 1]")
parser.add_argument("-kn", "--kernel_num", type=int, default=100)
parser.add_argument("-s", "--snapshot", type=str, default="models/best_steps_9500.pt")
parser.add_argument("-st", "--static", type=bool, default=True)
parser.add_argument("-m", "--middle_linear_size", type=int, default=7)
parser.add_argument("-o", "--class_num", type=int, default=81)

parser.add_argument("--vocab", type=str, default="text_fields.pt")
parser.add_argument("--PROJECT_NAME", type=str, default="TT Metro DWDM")
parser.add_argument("--BUSINESS_UNIT", type=str, default="1")
parser.add_argument("--REGION_ID", type=str, default="610823")
parser.add_argument("--REP_OFFICE_ID", type=str, default="2673")
parser.add_argument("--CUSTOMER_ID", type=str, default="50124")
parser.add_argument("--PROJECT_LEVEL_NAME", type=str, default="4")
parser.add_argument("--BUSINESS_GROUP_NAME", type=str, default="1")
parser.add_argument("--DELIVERY_TYPE", type=str, default="1")
parser.add_argument("--PROJECT_LABEL", type=str, default="1")
args = parser.parse_args()

text_fields = data.Field(sequential=True, lower=True)
label_fields = data.Field(sequential=False, use_vocab=False)

if args.vocab is None:
    raise ValueError("No vocabulary.")

text_fields.vocab = torch.load(args.vocab)

args.embed_num = len(text_fields.vocab)

if isinstance(args.kernel_sizes, list):
    kernel_sizes = [int(k) for k in args.kernel_sizes]
else:
    kernel_sizes = [int(k) for k in args.kernel_sizes[1:-1].split(',')]
args.kernel_sizes = kernel_sizes

net = TextCNN(args)

if args.snapshot is None:
    raise ValueError("No module, please train first.")

print("Loading model from {}...".format(args.snapshot))
net.load_state_dict(torch.load(args.snapshot))

net.eval()

FEATURE_LABEL = ["PROJECT_NAME", "BUSINESS_UNIT", "REGION_ID", "REP_OFFICE_ID",
                 "CUSTOMER_ID", "PROJECT_LEVEL_NAME", "BUSINESS_GROUP_NAME", "DELIVERY_TYPE", "PROJECT_LABEL"
                 ]

feature = []
for label in FEATURE_LABEL:
    text = getattr(args, label)
    text = text_fields.preprocess(text)
    text = [[text_fields.vocab.stoi[x] for x in text]]
    x = text_fields.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    feature.append(x.squeeze(0))

# feature = torch.cat(tuple(feature), 0)
for f in feature:
    f.data.t_()

print(feature)
exit(1)
output = net(x)
_, predicted = torch.max(output, 1)

label = label_fields.vocab.itos[predicted.data[0] + 1]

print("[Text]  {}\n[Label] {}\n".format(text, label))
