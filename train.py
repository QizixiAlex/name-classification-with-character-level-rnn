import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import init_data, load_data, NameDataset, all_categories, all_letters
from model import RNN


# parameters
n_hidden = 128
epochs = 10
plot_every = 1000
test_dataset_size = 100


def evaluate_test_dataset(rnn, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    total_correct = 0
    total = 0
    for idx, (category, name) in enumerate(test_dataloader):
        total += 1
        category, name = category[0], name[0]
        category, name = Variable(category), Variable(name)
        hidden = rnn.initHidden()
        for i in range(name.size()[0]):
            output, hidden = rnn(name[i], hidden)
        _, predicted = torch.max(output.data, 1)
        if predicted == category:
            total_correct += 1
    return total_correct/total


# init data
init_data()
# init model

rnn = RNN(len(all_letters), n_hidden, len(all_categories))
# setup data
train_data, test_data = load_data(test_dataset_size)
train_dataset = NameDataset(train_data)
test_dataset = NameDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# setup optimizer and criterion
optimizer = optim.Adam(rnn.parameters(), lr=0.0005)
criterion = nn.NLLLoss()
# train
all_loss = []
all_correct = []
for epoch in range(epochs):
    rnn = rnn.train()
    current_loss = 0
    for idx, (category, name) in enumerate(train_dataloader):
        category, name = category[0], name[0]
        category, name = Variable(category), Variable(name)
        optimizer.zero_grad()
        hidden = rnn.initHidden()
        for i in range(name.size()[0]):
            output, hidden = rnn(name[i], hidden)
        loss = criterion(output, category)
        loss.backward()
        optimizer.step()
        current_loss += loss
        if idx >= plot_every and idx % plot_every == 0:
            all_correct.append(evaluate_test_dataset(rnn, test_dataset))
            all_loss.append(float(current_loss)/plot_every)
            current_loss = 0
            rnn = rnn.train()

# plot all loss
plt.figure(0)
plt.plot(all_loss)
plt.figure(1)
plt.plot(all_correct)
plt.show()
# save model
torch.save(rnn, 'saved_model/char-rnn-classification.pt')