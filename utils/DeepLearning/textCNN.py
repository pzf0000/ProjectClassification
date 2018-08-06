import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Text_CNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(Text_CNN, self).__init__()
        self.static = static

        self.embed = nn.Embedding(embed_num, embed_dim)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, embed_dim)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        # self.conv13 = nn.Conv2d(1, kernel_num, (3, embed_dim))
        # self.conv14 = nn.Conv2d(1, kernel_num, (4, embed_dim))
        # self.conv15 = nn.Conv2d(1, kernel_num, (5, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)
        
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # x1 = self.conv_and_pool(x,self.conv13)
        # x2 = self.conv_and_pool(x,self.conv14)
        # x3 = self.conv_and_pool(x,self.conv15)
        # x = torch.cat((x1, x2, x3), 1)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
