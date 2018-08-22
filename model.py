import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


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