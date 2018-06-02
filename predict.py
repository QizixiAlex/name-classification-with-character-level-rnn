import sys
import torch
from torch.autograd import Variable
from data import all_categories, name_to_tensor, init_data

init_data()
rnn = torch.load('saved_model/char-rnn-classification.pt')


def evaluate_name(name):
    rnn.eval()
    name_tensor = Variable(name_to_tensor(name))
    hidden = rnn.initHidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)
    return output


def predict(name, n_predictions=3):
    output = evaluate_name(name)
    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = int(topi[0][i])
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])
    return predictions


if __name__ == '__main__':
    predict(sys.argv[1])