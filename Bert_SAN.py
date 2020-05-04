from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertForTokenClassification, BertModel
import torch, tqdm, json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
import random

class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size, drop_rate=0.1):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = torch.nn.Dropout(drop_rate)

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class BilinearFlatSim(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """

    def __init__(self, x_size, y_size, drop_rate=0.1):
        super(BilinearFlatSim, self).__init__()

        self.linear = nn.Linear(y_size, x_size)
        self.dropout = torch.nn.Dropout(drop_rate)

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)

        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class Classifier(nn.Module):
    def __init__(self, x_size, y_size, drop_rate=0.1):
        super(Classifier, self).__init__()

        self.dropout = torch.nn.Dropout(drop_rate)
        self.proj = nn.Linear(x_size * 4, y_size)

    def forward(self, x1, x2, mask=None):
        x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores

def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training: dropout_p = 0.0
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1)-1)
        new_data[i][one] = 1
    mask = 1.0/(1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask


class BertSAN(nn.Module):
    """BERT model with SAN for entailment.
    """

    def __init__(self, K, x_size, h_size, drop_rate=0.0, num_labels=3):
        super(BertSAN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.query_wsum = LinearSelfAttn(x_size, drop_rate=0.1)
        self.attn = BilinearFlatSim(x_size, h_size)
        self.rnn = torch.nn.GRUCell(input_size=768, hidden_size=768)

        self.K = K

        self.dropout = torch.nn.Dropout(drop_rate)
        self.classifier = Classifier(x_size, num_labels)

        self.alpha = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        hidden_states, pooled_output = self.bert(input_ids)

        x_mask = torch.BoolTensor((token_type_ids == 0) + (attention_mask == 0))
        h_mask = torch.BoolTensor((token_type_ids == 1) + (attention_mask == 0))

        h0 = x = hidden_states

        h0 = self.query_wsum(h0, h_mask)
        scores_list = []

        for turn in range(self.K):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)

            h0 = self.dropout(h0)
            h0 = self.rnn(x_sum, h0)

        mask = generate_mask(self.alpha.data.new(x.size(0), self.K), 0.1, self.training)
        mask = [m.contiguous() for m in torch.unbind(mask, 1)]
        tmp_scores_list = [mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1) for idx, inp in
                           enumerate(scores_list)]
        scores = torch.stack(tmp_scores_list, 2)
        scores = torch.mean(scores, 2)
        scores = torch.log(scores)

        #         scores = scores_list[-1]

        return scores
