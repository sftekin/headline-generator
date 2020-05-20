import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.seq2seq import Seq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(vocabs, batch_gen, train_params, model_params):
    word2int, int2word = vocabs
    num_epoch = train_params['num_epoch']
    learn_rate = train_params['learn_rate']
    tf_ratio = train_params['tf_ratio']
    clip = train_params['clip']
    eval_every = train_params['eval_every']

    net = Seq2Seq(vocab=int2word, **model_params).to(device)
    net.train()

    opt = optim.Adam(net.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=word2int['<pad>'])

    print('Training is starting ...')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epoch):
        running_loss = 0

        for idx, (x_cap, y_cap) in enumerate(batch_gen.generate('train')):
            print('\rtrain:{}/{}'.format(idx, batch_gen.num_iter('train')), flush=True, end='')
            x_cap, y_cap = x_cap.to(device), y_cap.to(device)

            opt.zero_grad()
            output = net(x_cap, y_cap, tf_ratio)

            loss = criterion(output.view(-1, output.size(2)), y_cap.view(-1).long())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            running_loss += loss.item()

            if (idx+1) % eval_every == 0:
                print('\n')
                val_loss = evaluate(net, word2int, batch_gen, tf_ratio)
                print("Epoch: {}/{}...".format(epoch + 1, num_epoch),
                      "Step: {}...".format(idx),
                      "Loss: {:.4f}...".format(running_loss / idx),
                      "Val Loss: {:.4f}\n".format(val_loss))

        # print('Creating sample captions')
        # sample(net, batch_gen, top_k=5, **kwargs)
        # print('\n')

        train_loss_list.append(running_loss / idx)
        val_loss_list.append(val_loss)

        loss_file = open('results/losses.pkl', 'wb')
        model_file = open('results/seq2seq.pkl', 'wb')
        pickle.dump([train_loss_list, val_loss_list], loss_file)
        pickle.dump(net, model_file)

    print('Training finished, saving the model')
    model_file = open('seq2seq.pkl', 'wb')
    pickle.dump(net, model_file)


def evaluate(net, vocab, batch_gen, tf_ratio):
    net.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    val_losses = []
    for idx, (x_cap, y_cap) in enumerate(batch_gen.generate('validation')):

        print('\reval:{}/{}'.format(idx, batch_gen.num_iter('validation')), flush=True, end='')

        x_cap, y_cap = x_cap.to(device), y_cap.to(device)
        output = net(x_cap, y_cap, tf_ratio)

        val_loss = criterion(output.view(-1, output.size(2)), y_cap.view(-1).long())
        val_losses.append(val_loss.item())

    net.train()
    return np.mean(val_losses)
