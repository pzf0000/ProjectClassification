import re
import random
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchtext import data
from utils.IO.IO import load_dataset

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
        """
        0   PROJECT_NAME
        1   BUSINESS_UNIT
        2   REGION_ID
        3   REP_OFFICE_ID
        4   CUSTOMER_ID
        5   PROJECT_LEVEL_NAME
        6   BUSINESS_GROUP_NAME
        7   DELIVERY_TYPE
        8   PROJECT_LABEL
        """
        fields = [("PROJECT_NAME", text_fields),
                  ("BUSINESS_UNIT", text_fields),
                  ("REGION_ID", text_fields),
                  ("REP_OFFICE_ID", text_fields),
                  ("CUSTOMER_ID", text_fields),
                  ("PROJECT_LEVEL_NAME", text_fields),
                  ("BUSINESS_GROUP_NAME", text_fields),
                  ("DELIVERY_TYPE", text_fields),
                  ("PROJECT_LABEL", text_fields)]

        from utils.DataPrepare.scenario import scenario_choice
        for c in scenario_choice:
            fields += [(c, label_fields)]

        if examples is None:
            examples = []
            for item in dataset:
                examples += [data.Example.fromlist(list(item), fields)]
        super(mydataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, dataset, text_field, label_field, dev_ratio=.2, test_ratio=.2, shuffle=True, **kwargs):
        examples = cls(dataset, text_fields, label_fields, **kwargs).examples

        if shuffle:
            random.shuffle(examples)

        dev_index = -1 * int((dev_ratio + test_ratio) * len(examples))
        test_index = -1 * int(test_ratio * len(examples))

        return (cls(dataset, text_field, label_field, examples=examples[:dev_index]),
                cls(dataset, text_field, label_field, examples=examples[dev_index:test_index]),
                cls(dataset, text_field, label_field, examples=examples[test_index:]))


class Config:
    batch_size = 64
    kernel_sizes = [3, 4, 5]
    kernel_num = 100
    # embed_num = 0
    embed_dim = 128
    # class_num = 0
    dropout = 0.5


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

        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.covs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

    def forward(self, x):
        x = self.embed(x)

        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.covs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit

net = TextCNN(args)
