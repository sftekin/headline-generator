import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.seq2seq import Seq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(vocabs, batch_gen, train_params, model_params):
    word2int, int2word = vocabs
    num_epoch = train_params['num_epoch']
    learn_rate = train_params['learn_rate']
    tf_ratio = train_params['tf_ratio']
    clip = train_params['clip']
    eval_every = train_params['eval_every']

    net = Seq2Seq(vocab=int2word, device=device, **model_params).to(device)
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
                print("\nEpoch: {}/{}...".format(epoch + 1, num_epoch),
                      "Step: {}...".format(idx),
                      "Loss: {:.4f}...".format(running_loss / idx),
                      "Val Loss: {:.4f}\n".format(val_loss))

        print('\nCreating sample captions')
        predict(net, vocabs, batch_gen.generate('validation'))
        print('\n')

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


def predict(net, vocabs, generator, tf_ratio=0.5, print_count=1, top_k=10):
    net.eval()
    word2int, int2word = vocabs
    criterion = nn.CrossEntropyLoss(ignore_index=word2int['<pad>'])

    outputs = []
    losses = []
    for x, y in generator:
        x, y = x.to(device), y.to(device)

        output = net(x, y, tf_ratio)
        outputs.append([y, output])

        loss = criterion(output.view(-1, output.size(2)), y.view(-1).long())
        losses.append(loss.item())

    if print_count > 0:
        translate(outputs[:print_count], int2word, top_k)

    net.train()
    return losses


def translate(outputs, int2word, top_k, remove_unk=True):
    for output in outputs:
        y_true, y_pre = output
        if torch.cuda.is_available():
            y_true, y_pre = y_true.cpu(), y_pre.cpu()
        batch_size = y_pre.shape[0]

        for i in range(batch_size):
            p = F.softmax(y_pre[i], dim=1).data

            p, top_ch = p.topk(top_k, dim=1)
            top_ch = top_ch.numpy()
            p = p.numpy()

            word_ints = []
            for j in range(len(p)):
                choice = np.random.choice(top_ch[j], p=p[j] / p[j].sum())
                word_ints.append(choice)

            pre_str = create_sen(word_ints, int2word, remove_unk=remove_unk)
            true_str = create_sen(y_true[i].numpy(), int2word, remove_unk=remove_unk)

            print("\nOriginal Title: {}\n"
                  "Predicted Title: {}".format(true_str, pre_str))


def create_sen(word_ints, vocab, remove_unk):
    output_str = ' '.join([vocab[tit] for tit in word_ints])
    output_str = output_str.replace('<end>', '').replace('<pad>', '')
    output_str = ' '.join(output_str.split())
    if remove_unk:
        output_str = output_str.replace('unk', '')
        output_str = ' '.join(output_str.split())

    return output_str