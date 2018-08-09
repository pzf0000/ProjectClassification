import os
import argparse
import datetime
import torch
from sklearn.model_selection import train_test_split
from torchtext import data
from utils.DeepLearning.mydatasets import Scenario
from utils.DeepLearning import training
from utils.DeepLearning.textCNN import Text_CNN
from utils.IO.IO import load_dataset


def scenario(dataset, index, text_field, label_field, batch_size, **kargs):
    train_data, dev_data, test_data = Scenario.splits(dataset, index, text_field, label_field)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                           batch_sizes=(batch_size, len(dev_data), len(test_data)),
                                                           **kargs)
    return train_iter, dev_iter, test_iter


def train_and_test(model,
                   model_name=None,
                   save_dir=None,
                   index_start=11,
                   index_end=91,
                   batch_size=64,
                   kernel_sizes="3,4,5",
                   embed_dim=128,
                   kernel_num=100,
                   dropout=0.5,
                   snapshot=None,
                   predict=None,
                   train=True,
                   test=True,
                   static=False):
    """
    :param batch_size: batch size for training [default: 64]
    :param kernel_sizes: comma-separated kernel size to use for convolution
    :param save_dir: where to save the snapshot
    :param embed_dim: number of embedding dimension [default: 128]
    :param kernel_num: number of each kind of kernel
    :param dropout: the probability for dropout [default: 0.5]
    :param snapshot: filename of model snapshot [default: None]
    :param predict: predict the sentence given
    :param train:
    :param test:
    :param model_name:
    :param static: fix the embedding
    :param index_start:
    :param index_end:
    :return:
    """
    dataset = load_dataset("../../data.npy")
    for index in range(index_start, index_end + 1):
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=True)
        train_iter, dev_iter, test_iter = scenario(dataset, index, text_field, label_field,
                                                   device=-1, repeat=False, batch_size=batch_size)

        # update args and print
        embed_num = len(text_field.vocab)
        class_num = len(label_field.vocab) - 1

        if isinstance(kernel_sizes, list):
            kernel_sizes = [int(k) for k in kernel_sizes]
        else:
            kernel_sizes = [int(k) for k in kernel_sizes.split(',')]
        # save_dir = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # model
        model = model(embed_num=embed_num,
                      embed_dim=embed_dim,
                      class_num=class_num,
                      kernel_num=kernel_num,
                      kernel_sizes=kernel_sizes,
                      dropout=dropout,
                      static=static)

        if snapshot is not None:
            if snapshot[-1] != "/":
                snapshot += "/"
            step = 1000
            while True:
                model_path = model_name + "_" + str(index) + "_best_steps_" + str(step) + ".pt"
                if os.path.exists(snapshot + model_path):
                    break
                else:
                    step -= 100
            print('\nLoading model from {}...'.format(model_path))
            model.load_state_dict(torch.load(snapshot + model_path))

        # train or predict
        if predict is not None:
            label = training.predict(predict, model, text_field, label_field)
            print('\n[Text]  {}\n[Label] {}\n'.format(predict, label))

        if train:
            try:
                training.train(train_iter, dev_iter, model, index, save_dir=save_dir, model_name=model_name)
            except KeyboardInterrupt:
                print('\n' + '-' * 89)
                print('Exiting from training early')

        if test:
            try:
                r = training.eval(test_iter, model)
                with open("result.txt", "a") as file:
                    file.write(str(index))
                    file.write('\t')
                    file.write(r[1])
                    file.write('\n')
            except Exception as e:
                with open("result.txt", "a") as file:
                    file.write(str(index))
                    file.write("\t")
                    file.write("Sorry. The test dataset doesn't exist.\n")


if __name__ == '__main__':
    # train_and_test(model=Text_CNN, model_name="text_cnn", save_dir="model/text_cnn")
    train_and_test(model=Text_CNN, model_name="text_cnn",
                   snapshot="model/text_cnn", train=False, test=False, predict="Turkmenistan Test Project ")
