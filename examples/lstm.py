# 09-07-2019;
"""
LSTM sequence model example, based on
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warm
import warm.functional as W


training_data = [
    ('The dog ate the apple'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    ('Everybody read that book'.split(), ['NN', 'V', 'DET', 'NN']), ]
testing_data = [('The dog ate the book'.split(), ['DET', 'NN', 'V', 'DET', 'NN'])]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}
ix_to_tag = {v:k for k, v in tag_to_ix.items()}


class WarmTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.arg = (embedding_dim, hidden_dim, vocab_size, tagset_size)
        warm.engine.prepare_model_(self, torch.tensor([0, 1], dtype=torch.long))
    def forward(self, x): # D
        embedding_dim, hidden_dim, vocab_size, tagset_size = self.arg
        y = W.embedding(x, embedding_dim, vocab_size) # D->DC
        y = W.lstm(y.T[None, ...], hidden_dim) # DC->BCD
        y = W.linear(y, tagset_size) # BCD
        y = F.log_softmax(y, dim=1) # BCD
        return y[0].T # DC


class TorchTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--warm', action='store_true', help='use warm instead of vanilla pytorch.')
    p = parser.parse_args()
    torch.manual_seed(1)
    #
    arg = (6, 6, len(word_to_ix), len(tag_to_ix))
    model = WarmTagger(*arg) if p.warm else TorchTagger(*arg)
    print(f'Using {model._get_name()}.')
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    #
    for epoch in range(300):
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    #
    with torch.no_grad():
        inputs = prepare_sequence(testing_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        ix = torch.argmax(tag_scores, -1).numpy()
        print(testing_data[0][0])
        print('Network tags:\n', [ix_to_tag[i] for i in ix])
        print('True tags:\n', testing_data[0][1])


if __name__ == '__main__':
    main()
