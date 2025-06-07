import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd # type: ignore
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from itertools import product
from tqdm import tqdm
import time


@dataclass
class Instance():
    def __init__(self, text, label):
        self.text = text.split(' ')
        self.label = label

class Vocab():
    def __init__(self, frequencies, max_size, min_freq, special_symbols=True):        
        if special_symbols:
            self.stoi = {'<PAD>': 0, '<UNK>': 1}
            self.itos = {0: '<PAD>', 1: '<UNK>'}
            self.n_spec_symbols = 2
        else:
            self.stoi = {}
            self.itos = {}
            self.n_spec_symbols = 0

        frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

        for i, (word, frequency) in enumerate(frequencies):
            if frequency >= min_freq and (len(self.stoi) < max_size or max_size == -1):
                self.stoi[word] = i+self.n_spec_symbols
                self.itos[i+self.n_spec_symbols] = word
            else:
                break


    def __getitem__(self, key):
        if isinstance(key, int):
            return self.itos[key]
        elif isinstance(key, str):
            return self.stoi[key]
        else:
            raise ValueError('key must be either int or str')
        
    def __len__(self):
        return len(self.stoi)
        
    def encode(self, text):
        if isinstance(text, str):
            return torch.tensor(self.stoi.get(text, 1))
        return torch.tensor([self.stoi.get(word, 1) for word in text])

    

class NLPDataset(Dataset):
    def __init__(self, path='data/sst_train_raw.csv', vocab=(None, None), max_size=-1, min_freq=1):
        self.instances = []
        csv = pd.read_csv(path, header=None)
        for _, row in csv.iterrows():
            self.instances.append(Instance(row[0], row[1]))

        if vocab[0] is not None:
            self.text_vocab, self.label_vocab = vocab
        else:
            word_frequencies = {}
            label_frequencies = {}
            for instance in self.instances:
                for word in instance.text:
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
                label_frequencies[instance.label] = label_frequencies.get(instance.label, 0) + 1

            self.text_vocab = Vocab(word_frequencies, max_size=max_size, min_freq=min_freq)
            self.label_vocab = Vocab(label_frequencies, max_size=max_size, min_freq=min_freq, special_symbols=False)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.text_vocab.encode(self.instances[idx].text), self.label_vocab.encode(self.instances[idx].label)


class Embedding(nn.Module):
    def __init__(self, vocabular, path=None, embedding_dim=300):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(len(vocabular), embedding_dim, padding_idx=vocabular['<PAD>'])

        emb = {}
        if path is not None:
            with open(path) as f:
                for line in f:
                    word, vec = line.split(' ', 1)
                    emb[word] = torch.tensor([float(x) for x in vec.split()])
        for i, word in vocabular.itos.items():
            if word in emb:
                self.embedding.weight.data[i] = emb[word]
            elif word == '<PAD>':
                self.embedding.weight.data[i] = torch.zeros(embedding_dim)
        self.embedding.freeze = path is not None

    def forward(self, x):
        return self.embedding(x)

    
def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_len = max(lengths)
    padded_texts = torch.stack([F.pad(text, (0, max_len - len(text)), value=pad_index) for text in texts])
    return padded_texts, torch.tensor(labels), torch.tensor(lengths)


class AveragePool(nn.Module):
    def __init__(self):
        super(AveragePool, self).__init__()

    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x
    

def train(model, data, optimizer, criterion, args):
    model.train()
    with tqdm(data, unit='batch') as t:
        for batch_num, batch in enumerate(t):
            optimizer.zero_grad()
            x, y, _ = batch
            logits = model(x).squeeze()
            loss = criterion(logits, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip"])
            optimizer.step()
        t.set_postfix(loss=loss.item())


def evaluate(model, data, criterion, args):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        cf_mat = torch.zeros(2, 2)
        for batch_num, batch in enumerate(data):
            x, y, _ = batch
            logits = model(x).squeeze()
            loss = criterion(logits, y.float())
            avg_loss += loss.item()
            cf_mat += confusion_matrix(y, logits > 0.5, labels=[0, 1])

        avg_loss /= len(data)
        acc = (cf_mat[0,0] + cf_mat[1,1]) / cf_mat.sum()
        f1 = 2 * cf_mat[1, 1] / (2*cf_mat[1, 1] + cf_mat[0, 1] + cf_mat[1, 0])
        # print(f'Validation loss: {avg_loss}')
        # print(f'Validation accuracy: {acc}')
        # print(f'Validation F1: {f1}')
        # print(f'Confusion matrix:\n{cf_mat.detach().numpy()}')
        return avg_loss, acc, f1

class RNN_Flexible(nn.Module):
    def __init__(self, type_, embedding, input_size, hidden_size, num_layers, bidirectional=False, dropout=0):
        super(RNN_Flexible, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = embedding
        if type_ == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        elif type_ == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        elif type_ == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError('type must be either rnn, lstm or gru')
        
        fc_input_size = hidden_size * (2 if bidirectional else 1)

        # Define the fully connected layers
        self.fc_1 = nn.Linear(fc_input_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, _ = self.rnn(x)

        # the model gives the output for every element of the sequence we are interested in the last one
        output = output[-1, :, :]
        x = self.fc_1(output)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

if __name__ == "__main__":

    args = {
        'seed': 7052020,
        'lr': [1e-3, 1e-4, 1e-5],
        'epochs': 6,
        'batch_size': 10,
        'clip': 0.25,
        'embedding_dim': 300,
        'vocab_size': -1,
        'min_freq': 1,
        'num_layers': [2, 3, 4],
        'hidden_size': [150, 200, 400],
        'rnn_type': ['gru', 'lstm'],
        'dropout': [0.2, 0.5, 0.8],
        'bidirectional': False,
        'optimizer': ['adam', 'rmsprop', 'sgd']
        
    }

    use_embedding = False
    try_base = False
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    train_set = NLPDataset('data/sst_train_raw.csv', max_size=-1)
    val_set = NLPDataset('data/sst_valid_raw.csv', vocab=(train_set.text_vocab, train_set.label_vocab))
    test_set = NLPDataset('data/sst_test_raw.csv', vocab=(train_set.text_vocab, train_set.label_vocab))

    
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)


    if use_embedding and args['embedding_dim'] == 300:
        embedding = Embedding(train_set.text_vocab, path='data/sst_glove_6b_300d.txt', embedding_dim=args['embedding_dim'])
    else:
        embedding = Embedding(train_set.text_vocab, embedding_dim=args['embedding_dim'])


    if try_base:
        print('Baseline model:')
        model = nn.Sequential(
                embedding,
                AveragePool(),
                nn.Linear(300, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 1)
            )
        loss = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        
        for epoch in range(args['epochs']):
            print('--------------------------------------------------------------------')
            print(f'epoch {epoch}')
            train(model, train_loader, optimizer, loss, args)
            evaluate(model, val_loader, loss, args)
        print('\nTest results for baseline model:')
        base_avg_loss, base_acc, base_f1 = evaluate(model, test_loader, loss, args)

    
    # results_table = pd.DataFrame([], columns=['rnn_type', 'lr', 'num_layers', 'hidden_size', 'dropout', 'optimizer', 'loss', 'acc', 'f1', 'training_time'])
    results_table = pd.read_csv('exp/shootout_2.csv')
    combinations = list(product(args['rnn_type'], args['lr'], args['num_layers'], args['hidden_size'], args['dropout'], args['optimizer']))
    for i, comb in enumerate(combinations):
        
        args['rnn_type'], args['lr'], args['num_layers'], args['hidden_size'], args['dropout'], args['optimizer'] = comb

        index = (
            (results_table['rnn_type'] == args['rnn_type']) &
            (results_table['lr'] == args['lr']) &
            (results_table['num_layers'] == args['num_layers']) &
            (results_table['hidden_size'] == args['hidden_size']) &
            (results_table['dropout'] == args['dropout']) &
            (results_table['optimizer'] == args['optimizer'])
        )

        
        if index.sum() > 0:
            print('already checked')
            continue

        model = RNN_Flexible(args['rnn_type'], embedding, 300, args['hidden_size'], args['num_layers'],  dropout=args['dropout'])

        loss_fn = nn.BCEWithLogitsLoss()
        if args['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        elif args['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args['lr'])
        elif args['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args['lr'])
        print(f'Model: {args["rnn_type"]}')
        print(f'learning rate: {args["lr"]}')
        print(f'num_layers: {args["num_layers"]}')
        print(f'hidden_size: {args["hidden_size"]}')
        print(f'dropout: {args["dropout"]}')
        print(f'optimizer: {args["optimizer"]}')
        start = time.time()
        for epoch in range(args['epochs']):
            print('--------------------------------------------------------------------')
            print(f'epoch {epoch}')
            train(model, train_loader, optimizer, loss_fn, args)
            evaluate(model, val_loader, loss_fn, args)
        # print('--------------------------------------------------------------------')
        end = time.time()
        training_time = end - start

        # print(f'\nTest results for model {args["rnn_type"]}:')
        loss, acc, f1 = evaluate(model, test_loader, loss_fn, args)
        print('\n')

        results_table.loc[len(results_table)] = {
            'rnn_type': args['rnn_type'],
            'lr': args['lr'],
            'num_layers': args['num_layers'],
            'hidden_size': args['hidden_size'],
            'dropout': args['dropout'],
            'optimizer': args['optimizer'],
            'loss': loss,
            'acc': acc.item(),
            "f1": f1.item(),
            "training_time": training_time
        }

        results_table.to_csv('exp/shootout_2.csv',index=False)
    