import torch
import glob
import string
import unicodedata
from random import shuffle
from torch.utils.data import Dataset


# constants
names_file_path = 'names_data/names/*.txt'
all_letters = string.ascii_letters + " .,;'"
all_categories = []


# helper functions
# find all files in path
def find_files(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line).lower() for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Find category index from all indexes
def category_to_index(category):
    return all_categories.index(category)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def name_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


# Turn a name category to tensor
def category_to_tensor(category):
    return torch.tensor([all_categories.index(category)], dtype=torch.long)


def load_data(test_size):
    data = []
    for filename in find_files(names_file_path):
        category = filename.split('/')[-1].split('.')[0]
        lines = read_lines(filename)
        data = data + [(category, line) for line in lines]
    shuffle(data)
    test_data = data[:test_size]
    train_data = data[test_size:]
    return train_data, test_data


class NameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        category, name = self.data[index]
        return category_to_tensor(category), name_to_tensor(name)


def init_data():
    # init categories
    for filename in find_files(names_file_path):
        all_categories.append(filename.split('/')[-1].split('.')[0])

