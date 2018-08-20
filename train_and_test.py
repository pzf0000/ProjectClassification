import os
import re
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchtext import data

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-d", "--dropout", type=float, default=0.2)
parser.add_argument("-data", "--dataset", type=str, default="data.npy")
parser.add_argument("-e", "--epochs", type=int, default=256)
parser.add_argument("-ed", "--embed_dim", type=int, default=128)
parser.add_argument("-ks", "--kernel_sizes", type=str, default="[3, 4, 5]")
parser.add_argument("-kn", "--kernel_num", type=int, default=100)
parser.add_argument("-l", "--log_interval", type=int, default=1)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-s", "--save_interval", type=int, default=500)
parser.add_argument("-sd", "--save_dir", type=str, default="model")
parser.add_argument("-st", "--static", type=bool, default=False)
parser.add_argument("-t", "--test_interval", type=int, default=100)
parser.add_argument("-m", "--middle_linear_size", type=int, default=8)
parser.add_argument("-o", "--class_num", type=int, default=81)
args = parser.parse_args()


def load_dataset(filename=args.data):
    try:
        return np.load(filename)
    except:
        try:
            return pd.read_csv(filename)
        except:
            raise ValueError("Can not open file.")


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

dataset = load_dataset(args.data)

text_fields = data.Field(sequential=True, lower=True)
label_fields = data.Field(sequential=False, use_vocab=False)


class mydataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        """
        提供迭代策略
        :param ex:
        :return:
        """
        return len(ex.BUSINESS_GROUP_NAME)

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
        """
        将数据分为训练集、验证集和测试集
        :param dataset: 数据集
        :param text_field:
        :param label_field:
        :param dev_ratio: 验证集比例
        :param test_ratio: 测试集比例
        :param kwargs:
        :return: tuple
        """
        examples = cls(dataset, text_fields, label_fields, **kwargs).examples

        dev_index = -1 * int((dev_ratio + test_ratio) * len(examples))
        test_index = -1 * int(test_ratio * len(examples))

        return (cls(dataset, text_field, label_field, examples=examples[:dev_index]),
                cls(dataset, text_field, label_field, examples=examples[dev_index:test_index]),
                cls(dataset, text_field, label_field, examples=examples[test_index:]))


print("=========================\nParameters\n=========================\n")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


train_data, dev_data, test_data = mydataset.splits(dataset, text_fields, label_fields)
text_fields.build_vocab(train_data, dev_data)
label_fields.build_vocab(train_data, dev_data)
train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                       batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                                       device=-1, repeat=False)

args.embed_num = len(text_fields.vocab)


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.static = args.static
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.embed_dim)) for K in args.kernel_sizes])
        self.lin1 = nn.Linear(8, args.middle_linear_size)
        self.lin2 = nn.Linear(len(args.kernel_sizes) * args.kernel_num + args.middle_linear_size, len(args.kernel_sizes) * args.kernel_num)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

    def forward(self, x):
        x1 = self.embed(x[0])
        x2 = torch.cat(tuple([x[i] for i in np.arange(1, 9, 1)]), 1)
        x2 = x2.float()

        if self.static:
            x1 = Variable(x1)
            x2 = Variable(x2)

        # 使用卷积处理项目名称
        x1 = x1.unsqueeze(1)
        x1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs]
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x1]
        x1 = torch.cat(x1, 1)
        x1 = self.dropout(x1)

        x2 = F.relu(self.lin1(x2))

        # 将x1和x2合并
        x = torch.cat((x1, x2), 1)

        x = F.relu(self.lin2(x))

        logit = F.sigmoid(self.fc1(x))
        return logit


net = TextCNN(args)
print("=========================\nModule\n=========================\n")
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

steps = 0
best_acc = 0
last_step = 0
net.train()


def save(model, save_dir, save_prefix, steps, model_name=None):
    if save_dir is None:
        return

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if model_name is not None:
        save_prefix = model_name + "_" + save_prefix

    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def eval(data_iter, model):
    model.eval()
    corrects = 0.0
    avg_loss = 0
    for batch in data_iter:
        feature = []
        target = []
        for b in batch.fields:
            if b.isdigit():
                target.append(getattr(batch, b))
            else:
                feature.append(getattr(batch, b))
        for f in feature:
            f.data.t_()

        target_len = len(target)
        batch_len = batch.batch_size

        new_target_items = []
        for b in range(batch_len):
            new_item = torch.Tensor(list(target[t][b].tolist() for t in range(target_len)))
            new_target_items.append(new_item.reshape(1, len(new_item)))

        target = torch.cat(tuple(new_target_items), 0)

        logit = model(feature)

        loss = (((logit - target) ** 2).sum().float().requires_grad_(True) / batch_len) ** (1 / 2)

        avg_loss += loss.data[0]

        for i in range(batch_len):
            item_len = len(logit[i])
            correct = 0
            sum = 0
            for j in range(item_len):
                if target[i][j].int().data == 1 or logit[i][j].int().data == 1:
                    sum += 1
                    if logit[i][j].int() == target[i][j].int():
                        correct += 1
            corrects += float(correct) / float(sum)

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100 * corrects / size
    print("Evaluation [{}/{}] loss: {:.6f}  acc: {:.4f}%".format(int(corrects), size, avg_loss, accuracy))
    return accuracy


print("=========================\nTraining\n=========================")

for epoch in range(1, args.epochs + 1):
    losses = []
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

        # 81*64 --> 64*81
        # 将[标签][批次]改为[批次][标签]
        target_len = len(target)
        batch_len = batch.batch_size

        new_target_items = []
        for b in range(batch_len):
            new_item = torch.Tensor(list(target[t][b].tolist() for t in range(target_len)))
            new_target_items.append(new_item.reshape(1, len(new_item)))

        target = torch.cat(tuple(new_target_items), 0)

        optimizer.zero_grad()  # 清空所有优化的梯度
        logit = net(feature)

        # 需要torch.LongTensor
        # loss_func = nn.MultiLabelMarginLoss()
        # loss = loss_func(logit, target.long())
        # loss = ((logit - target) ** 2).sum() / (logit.shape[0] * logit.shape[1])
        # delta = logit - target
        # loss = ((delta ** 2).sum() / batch_len) ** (1 / 2)
        loss = (((logit - target) ** 2).sum().float().requires_grad_(True) / batch_len) ** (1 / 2)

        loss.backward()
        optimizer.step()

        steps += 1

        losses.append(loss.data.mean())

        if steps % args.test_interval == 0:
            dev_acc = eval(dev_iter, net)
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = steps
                save(net, args.save_dir, 'best', steps)
        elif steps % args.save_interval == 0:
            save(net, args.save_dir, 'snapshot', steps)

    print("Training [{}/{}] loss: {:.6f}".format(epoch, args.epochs, np.mean(losses)))

print("=========================\nTesting\n=========================")
acc = eval(test_iter, net)
