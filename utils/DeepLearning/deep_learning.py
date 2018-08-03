import os
import argparse
import datetime
import torch
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext import datasets
from utils.DeepLearning import mydatasets, textCNN, train
from utils.IO.IO import load_dataset


def arg_parser():
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    # parser.add_argument('-no_cuda', action='store_true', default=True, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-train', action='store_true', default=False, help='train')
    parser.add_argument('-test', action='store_true', default=False, help='test')
    args = parser.parse_args()
    return args


# load SST dataset
def sst(text_field, label_field, **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(
    args.batch_size, len(dev_data), len(test_data)), **kargs)
    return train_iter, dev_iter, test_iter


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits((train_data, dev_data), batch_sizes=(args.batch_size, len(dev_data)),
                                                **kargs)
    return train_iter, dev_iter


# load scenario dataset
def scenario(text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets.Scenario.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                           batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                                           **kargs)
    return train_iter, dev_iter, test_iter


def prepare_data(data, index, cv=False):
    feature_set = data[:, 0]
    label = data[:, index]
    if cv:
        x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)
    else:
        x_train = feature_set
        y_train = label

    file_pos = open("train.pos", "w")
    file_neg = open("train.neg", "w")

    len_train = len(x_train)
    for i in range(len_train):
        if y_train[i] == "1":
            file_pos.write(x_train[i])
            file_pos.write("\n")
        else:
            file_neg.write(x_train[i])
            file_neg.write("\n")

    file_pos.close()
    file_neg.close()

    if cv:
        file_pos = open("test.pos", "w")
        file_neg = open("test.neg", "w")

        len_test = len(x_test)
        for i in range(len_test):
            if y_test[i] == "1":
                file_pos.write(x_test[i])
                file_pos.write("\n")
            else:
                file_neg.write(x_test[i])
                file_neg.write("\n")

        file_pos.close()
        file_neg.close()


if __name__ == '__main__':
    # load data
    dataset = load_dataset("../data.npy")
    for index in range(11, 92):
        args = arg_parser()
        prepare_data(dataset, index)
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        # train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
        # train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)
        train_iter, dev_iter, test_iter = scenario(text_field, label_field, device=-1, repeat=False)

        # update args and print
        args.embed_num = len(text_field.vocab)
        args.class_num = len(label_field.vocab) - 1
        # args.cuda = (not args.no_cuda) and torch.cuda.is_available()
        # del args.no_cuda
        if isinstance(args.kernel_sizes, list):
            args.kernel_sizes = [int(k) for k in args.kernel_sizes]
        else:
            args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
        args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # print("\nParameters:")
        # for attr, value in sorted(args.__dict__.items()):
        #     print("\t{}={}".format(attr.upper(), value))

        # model
        cnn = textCNN.CNN_Text(args)
        if args.snapshot is not None:
            print('\nLoading model from {}...'.format(args.snapshot))
            cnn.load_state_dict(torch.load(args.snapshot))

        # if args.cuda:
        #     torch.cuda.set_device(args.device)
        #     cnn = cnn.cuda()

        # train or predict
        if args.predict is not None:
            label = train.predict(args.predict, cnn, text_field, label_field)
            print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))

        if args.train:
            try:
                train.train(train_iter, dev_iter, cnn, index, args)
            except KeyboardInterrupt:
                print('\n' + '-' * 89)
                print('Exiting from training early')

        if args.test:
            try:
                print()
                print()
                r = train.eval(test_iter, cnn, args)
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
