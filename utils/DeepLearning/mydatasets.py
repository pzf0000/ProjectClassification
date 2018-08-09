import re
import os
import random
import tarfile
import urllib
from torchtext import data


class Scenario(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, dataset, index, text_field, label_field, path=None, examples=None, **kwargs):

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

        label_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []

            for item in dataset:
                if item[index] == '0':
                    examples += [
                        data.Example.fromlist([item[0], "negative"], fields)]

                if item[index] == '1':
                    examples += [
                        data.Example.fromlist([item[0], "positive"], fields)]

        super(Scenario, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, dataset, index, text_field, label_field, dev_ratio=.1, test_ratio=.1, shuffle=True, root='.', **kwargs):
        path = root
        examples = cls(dataset, index, text_field, label_field, path=path, **kwargs).examples

        if shuffle:
            random.shuffle(examples)

        dev_index = -1 * int((dev_ratio+test_ratio) * len(examples))
        test_index = -1 * int(test_ratio * len(examples))

        return (cls(dataset, index, text_field, label_field, examples=examples[:dev_index]),
                cls(dataset, index, text_field, label_field, examples=examples[dev_index:test_index]),
                cls(dataset, index, text_field, label_field, examples=examples[test_index:]))
