import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, index,
          learning_rate=0.001,
          epochs=256,
          log_interval=1,
          test_interval=100,
          save_best=True,
          save_dir="model",
          save_interval=500,
          early_stop=1000,
          model_name=None):
    """
    训练
    :param train_iter: 训练集
    :param dev_iter: 验证集
    :param model: 模型
    :param index: 场景索引[11,91]
    :param learning_rate: 学习率[default: 0.001]
    :param epochs: 批次[default: 256]
    :param log_interval: 每隔多少输出一次结果[default: 1]
    :param test_interval: 测试间隔[default: 100]
    :param save_best: 存储最佳模型[default: True]
    :param save_dir: 存储目录
    :param save_interval: 存储间隔[default: 500]
    :param early_stop: 误差上升[default: 1000]
    :param model_name: 模型名称
    :return: None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # 批量优先，索引对齐

            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                if steps % 100 == 0:
                    print(str(index) + "\tBatch[{}]\t".format(steps) +
                          "loss: {:.6f}\tacc: {:.4f}%({}/{})".format(
                              loss.data[0], accuracy, corrects, batch.batch_size))

            if steps % test_interval == 0:
                dev_acc = eval(dev_iter, model)[0]
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if save_best:
                        save(model, save_dir, 'best', steps, index, model_name)
                else:
                    if steps - last_step >= early_stop:
                        print('early stop by {} steps.'.format(early_stop))
            elif steps % save_interval == 0:
                save(model, save_dir, 'snapshot', steps, index, model_name)


def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    s = 'Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                     accuracy,
                                                                     corrects,
                                                                     size)
    print(s)
    return accuracy, s


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()

    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)

    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)

    return label_feild.vocab.itos[predicted.data[0] + 1]


def save(model, save_dir, save_prefix, steps, index, model_name=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_prefix = str(index) + "_" + save_prefix
    if model_name is not None:
        save_prefix = model_name + "_" + save_prefix

    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
