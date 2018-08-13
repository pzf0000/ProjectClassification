import re
import random
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchtext import data
from utils.IO.IO import load_dataset

FEATURES = {
    "PROJECT_NAME": 0,
    "BUSINESS_UNIT": 1,
    "REGION_ID": 2,
    "REP_OFFICE_ID": 3,
    "CUSTOMER_ID": 4,
    "PROJECT_LEVEL_NAME": 5,
    "BUSINESS_GROUP_NAME": 6,
    "DELIVERY_TYPE": 7,
    "PROJECT_LABEL": 8
}
dataset = load_dataset("../../data.npy")

text_fields = data.Field(sequential=True, lower=True)
label_fields = data.Field(sequential=False, use_vocab=False)


class mydataset(data.Dataset):
    def __init__(self, dataset, text_fields, label_fields, examples=None, **kwargs):
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_fields.preprocessing = data.Pipeline(clean_str)
        fields = [(f, text_fields) for f in FEATURES]

        from utils.DataPrepare.scenario import scenario_choice
        for c in scenario_choice.values():
            fields.append((str(c), label_fields))

        if examples is None:
            examples = []
            for item in dataset:
                examples += [data.Example.fromlist(list(item), fields)]

        super(mydataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, dataset, text_field, label_field, dev_ratio=.2, test_ratio=.2, **kwargs):
        examples = cls(dataset, text_fields, label_fields, **kwargs).examples

        dev_index = -1 * int((dev_ratio + test_ratio) * len(examples))
        test_index = -1 * int(test_ratio * len(examples))

        return (cls(dataset, text_field, label_field, examples=examples[:dev_index]),
                cls(dataset, text_field, label_field, examples=examples[dev_index:test_index]),
                cls(dataset, text_field, label_field, examples=examples[test_index:]))


class Config:
    batch_size = 64
    kernel_sizes = [3, 4, 5]
    kernel_num = 100
    embed_dim = 128
    dropout = 0.5
    learning_rate = 0.001
    epochs = 256
    static = False
    log_interval = 1


args = Config()

train_data, dev_data, test_data = mydataset.splits(dataset, text_fields, label_fields)
text_fields.build_vocab(train_data, dev_data, test_data)
label_fields.build_vocab(train_data, dev_data, test_data)
train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                       batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                                       device=-1, repeat=False)

args.embed_num = len(text_fields.vocab)
args.class_num = len(label_fields.vocab) - 1


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.static = args.static
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.covs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

    def forward(self, x):
        x1 = self.embed(x[0])
        x2 = torch.cat(tuple([x[i] for i in np.arange(1, 9, 1)]), 1)
        # x2应为float类型，且应该有第三维度，第三维度size为128（即x1的embed_dim）
        x2 = x2.float()

        if self.static:
            x1 = Variable(x1)
            x2 = Variable(x2)

        x = torch.cat((x1, x2), 1)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.covs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


net = TextCNN(args)

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

steps = 0
best_acc = 0
last_step = 0
net.train()

for epoch in range(1, args.epochs + 1):
    for batch in train_iter:
        feature = []
        target = []
        for b in batch.fields:
            if b.isdigit():
                target.append(getattr(batch, b))
            else:
                feature.append(getattr(batch, b))
        for f in feature:
            f.data.t_()
        for t in target:
            t.data.sub_(1)

        optimizer.zero_grad()
        logit = net(feature)

        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

        steps += 1

        if steps % args.log_interval == 0:
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects / batch.batch_size
            if steps % 100 == 0:
                print("\tBatch[{}]\t".format(steps) +
                      "loss: {:.6f}\tacc: {:.4f}%({}/{})".format(
                          loss.data[0], accuracy, corrects, batch.batch_size))
