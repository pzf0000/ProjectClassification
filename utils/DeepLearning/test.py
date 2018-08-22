import os
import re
import time
import numpy as np
import datetime
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchtext import data
from utils.IO.IO import load_dataset
from utils.DataPrepare.scenario import scenario_choice

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
dataset = load_dataset("../../data2.npy")
# new_dataset = []
# for i in range(len(dataset)):
#     item = list(dataset[i, 0:9]) + list([dataset[i, scenario_choice["Wireless (Including TK)"]]])
#     item = item + list(dataset[i, scenario_choice["Fixed network"]])
#     item = item + list(dataset[i, scenario_choice["Energy"]])
#     item = item + list(dataset[i, scenario_choice["POD (Pure Equipment)"]])
#     item = item + list(dataset[i, scenario_choice["IBS"]])
#     item = item + list(dataset[i, scenario_choice["Software Integration Service"]])
#     new_dataset.append(item)
#
# dataset = np.array(new_dataset)


text_fields = data.Field(sequential=True, lower=True)
label_fields = data.Field(sequential=False, use_vocab=False)


class Config:
    batch_size = 64
    kernel_sizes = [1, 2, 3, 3, 2, 1]
    kernel_num = 100
    embed_dim = 128
    dropout = 0.5
    learning_rate = 0.002
    epochs = 256
    static = True
    log_interval = 1
    test_interval = 100
    save_interval = 500
    middle_linear_size = 7

    def __str__(self):
        s = str(self.__class__.__name__)
        s += "\n"
        for attr in sorted(self.__dir__()):
            if attr[0] != "_":
                s += "{} = {}\n".format(attr, getattr(self, attr))
        return s


class mydataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
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

        for c in scenario_choice.values():
            fields.append((str(c), label_fields))

        # fields.append((str(scenario_choice["Wireless (Including TK)"]), label_fields))
        # fields.append((str(scenario_choice["Fixed network"]), label_fields))
        # fields.append((str(scenario_choice["Energy"]), label_fields))
        # fields.append((str(scenario_choice["POD (Pure Equipment)"]), label_fields))
        # fields.append((str(scenario_choice["IBS"]), label_fields))
        # fields.append((str(scenario_choice["Software Integration Service"]), label_fields))

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


args = Config()

file_name = str(datetime.datetime.now())[:19]
file_name = file_name.replace(' ', '_')
file_name = file_name.replace('-', '_')
file_name = file_name.replace(':', '_')
file = open(file_name + ".txt", "w")

s = "=========================\nParameters\n=========================\n"

s += str(args)
print(s)
file.write(s)
file.write("\n")


train_data, dev_data, test_data = mydataset.splits(dataset, text_fields, label_fields)
text_fields.build_vocab(train_data, dev_data)
label_fields.build_vocab(train_data, dev_data)
train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                       batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                                       device=-1, repeat=False)

args.embed_num = len(text_fields.vocab)
# args.class_num = len(label_fields.vocab) - 1
args.class_num = 81


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
s = "=========================\nModule\n=========================\n"
s += str(net)
print(s)
file.write(s)
file.write('\n')

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

steps = 0
best_acc = 0
last_step = 0
net.train()


def save(model, save_dir, save_prefix, steps, model_name=None):
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

        # 使用距离的平方作为loss，最后除以每一个batch的大小
        # delta = logit - target
        # loss = ((delta ** 2).sum() / batch_len) ** (1 / 2)
        # loss = (logit.int() == target.int()).sum() / batch_len
        loss = (((logit - target) ** 2).sum().float().requires_grad_(True) / batch_len) ** (1 / 2)

        avg_loss += loss.data[0]

        for i in range(batch_len):
            item_len = len(logit[i])
            correct = 0
            sum = 0
            for j in range(item_len):
                # if target[i][j].int().data == 1:
                if target[i][j].int().data == 1 or logit[i][j].int().data == 1:
                    sum += 1
                    if logit[i][j].int() == target[i][j].int():
                        correct += 1
            if sum != 0:
                corrects += float(correct) / sum

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100 * corrects / size
    s = "Evaluation [{}/{}] loss: {:.6f}  acc: {:.4f}%".format(int(corrects), size, avg_loss, accuracy)
    print(s)
    file.write(s)
    file.write('\n')
    return accuracy


start = time.time()
start_str = "=========================\nTraining\n========================="
print(start_str)
file.write(start_str)
file.write('\n')

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
        #         save(net, "model", 'best', steps)
        # elif steps % args.save_interval == 0:
        #     save(net, "model", 'snapshot', steps)

    s = "Training [{}/{}] loss: {:.6f}".format(epoch, args.epochs, np.mean(losses))
    print(s)
    file.write(s)
    file.write('\n')

end = time.time()
s = "Training spent: {:.2f} min".format((end - start) / 60)
print(s)
file.write(s)
file.write('\n')

start_str = "=========================\nTesting\n========================="
print(start_str)
file.write(start_str)
file.write('\n')

start = time.time()
acc = eval(test_iter, net)
end = time.time()
s = "Testing spent: {:.2f} ms".format((end - start) * 1000)
print(s)
file.write(s)
file.write('\n')

file.close()

with open("results.txt", "a+") as f:
    f.write(str(args.dropout) + "\t" + str(args.middle_linear_size) + "\t" + str(args.kernel_num) + "\t")
    f.write(str(args.learning_rate) + "\t" + str(acc) + "%\t" + str(best_acc) + "%\t" + str((end - start) * 1000))
    f.write("\t" + str(args.epochs))
    f.write("\n")
