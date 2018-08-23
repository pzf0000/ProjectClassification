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
parser.add_argument("-ml", "--machine_learning_model", type=str, default="ml_model.npy")
parser.add_argument("-mp", "--machine_learning_proportion", type=float, default=0.5)
parser.add_argument("-o", "--class_num", type=int, default=81)
parser.add_argument("-l", "--label", type=bool, default=False)
parser.add_argument("-v", "--vocab", type=str, default="text_fields.pt")

parser.add_argument("--PROJECT_NAME", type=str, default="Italy Fastweb Optical&Access Building Switch Project")
parser.add_argument("--BUSINESS_UNIT", type=str, default="1")
parser.add_argument("--REGION_ID", type=str, default="60666")
parser.add_argument("--REP_OFFICE_ID", type=str, default="2686")
parser.add_argument("--CUSTOMER_ID", type=str, default="29679565")
parser.add_argument("--PROJECT_LEVEL_NAME", type=str, default="4")
parser.add_argument("--BUSINESS_GROUP_NAME", type=str, default="1")
parser.add_argument("--DELIVERY_TYPE", type=str, default="0")
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

output = None

FEATURE_LABEL = ["PROJECT_NAME", "BUSINESS_UNIT", "REGION_ID", "REP_OFFICE_ID",
                 "CUSTOMER_ID", "PROJECT_LEVEL_NAME", "BUSINESS_GROUP_NAME", "DELIVERY_TYPE", "PROJECT_LABEL"
                 ]

# Deep Learning
if args.snapshot is not None:
    print("Loading model from {}...".format(args.snapshot))
    net.load_state_dict(torch.load(args.snapshot))

    net.eval()
    feature = []
    for label in FEATURE_LABEL:
        text = getattr(args, label)
        text = text_fields.preprocess(text)
        text = [[text_fields.vocab.stoi[x] for x in text]]
        x = text_fields.tensor_type(text)
        x = autograd.Variable(x, volatile=True)
        feature.append(x)

    dl_output = net(feature).int().squeeze(0).tolist()

# Machine Learning
if args.machine_learning_model is not None:
    classifiers = np.load(args.machine_learning_model)

    feature = []
    for label in FEATURE_LABEL:
        text = getattr(args, label)
        if text.isdigit():
            feature.append(int(text))
    feature = np.array(feature)
    feature = feature.reshape(1, -1)

    results = []
    for c in classifiers:
        result = c.predict(feature)
        results.append(result.toarray())

    result = [[0] * len(results[0][0])] * len(results[0])
    weight = [1] * len(results[0])
    # 根据 weight 相加
    for i in range(len(results)):
        result += results[i] * weight[i]
    ml_output = result[0].round()


if output is None:
    raise ValueError("No output, maybe you need to train a model first.")

if args.label:
    from prepare_data_from_csv import scenario_choice

    key_list = []
    value_list = []

    for key, value in scenario_choice.items():
        key_list.append(key)
        value_list.append(value)

    for i in range(len(output)):
        if output[i] == 1:
            if i in value_list:
                get_value_index = value_list.index(i)
                print(key_list[get_value_index])
            else:
                raise KeyError("The scenario code is {}. It not in the dictionary.".format(str(i)))
else:
    print(output)
