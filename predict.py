
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


label = predict(predict, model, text_field, label_field)
print('\n[Text]  {}\n[Label] {}\n'.format(predict, label))