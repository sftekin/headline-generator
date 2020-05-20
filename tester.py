import torch
import numpy as np

from transformers.bleu import compute_bleu
from trainer import translate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, vocab, batch_gen):
    net.eval()
    net.to(device)

    bleu_all_score = []
    for idx, (content, title) in enumerate(batch_gen.generate('test')):

        print('\rtest:{}'.format(idx), flush=True, end='')

        content, title = content.to(device), title.to(device)
        title_pre = net(content, title, tf_ratio=0.1)

        referance_caps, translate_caps = translate(outputs=[title, title_pre],
                                                   int2word=vocab,
                                                   top_k=10,
                                                   print_count=10)
        print('\n')
        for i in range(len(referance_caps)):
            bleu_score_list = []
            for j in range(1, 5):
                bleu, _, _ = compute_bleu(referance_caps[i], translate_caps[i], max_order=j)
                bleu_score_list.append(bleu)
            bleu_all_score.append(bleu_score_list)

    bleu_all_score = np.array(bleu_all_score)
    bleu_score_arr = np.mean(bleu_all_score, axis=0)

    for i in range(4):
        print("BLUE {}: {:.2f}".format(i+1, bleu_score_arr[i]))

