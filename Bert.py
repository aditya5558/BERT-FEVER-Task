import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

class Config():
    
    def __init__(self):
        
        self.eps = 1e-12
        
        # Defining the maximum number of the token, embeddings and positions
        self.vocab_size = 30522
        self.sent_size = 2
        self.pos_size = 512
        
        # Defining the embedding dimensions of the tokens, sentences and positions
        self.emb_dim = 768
        
        # Embedding layer dropout rate
        self.emb_drop_rate = 0.1
        
        # Attention Layer
        self.num_heads = 12
        self.attn_drop_rate = 0.1
        
        # FeedForward Lyaer
        self.hidden_dim = 3072
        self.fc_drop_rate = 0.1
        
        # Encoder
        self.num_encoders = 12

class EmbeddingLayer(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        self.token_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.emb_dim)
        self.sentence_embeddings = nn.Embedding(num_embeddings=config.sent_size, embedding_dim=config.emb_dim)
        self.positional_embeddings = nn.Embedding(num_embeddings=config.pos_size, embedding_dim=config.emb_dim)
        
        self.layer_norm = nn.LayerNorm(normalized_shape=config.emb_dim, eps=config.eps)
        self.emb_dropout = nn.Dropout(p=config.emb_drop_rate)
        
    def forward(self, token_ip, sent_ip, pos_ip):
        
        token_emb = self.token_embeddings(token_ip)
        sent_emb = self.sentence_embeddings(sent_ip)
        pos_emb = self.positional_embeddings(pos_ip)
        
        embeddings = token_emb + sent_emb + pos_emb
        
        normalized_emb = self.layer_norm(embeddings)
        output = self.emb_dropout(normalized_emb)
        
        return output

class MultiHeadSelfAttentionLayer(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        #config.num_heads should divide config.emb_dim
        self.config = config
        self.each_head_dim = config.emb_dim//config.num_heads
        
        self.queries = nn.Linear(in_features=config.emb_dim, out_features=config.emb_dim)
        self.keys = nn.Linear(in_features=config.emb_dim, out_features=config.emb_dim)
        self.values = nn.Linear(in_features=config.emb_dim, out_features=config.emb_dim)

        self.attn_dropout = nn.Dropout(p=config.attn_drop_rate)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.linear = nn.Linear(in_features=config.emb_dim, out_features=config.emb_dim)
        
    
    def forward(self, X, mask=None):
        
        Q = self.queries(X)
        K = self.keys(X)
        V = self.values(X)
        
        old_shape = list(Q.shape)
        new_shape = old_shape[:-1] + [self.config.num_heads, self.each_head_dim]
        
        Q = torch.transpose(torch.reshape(Q, new_shape), 1, 2)
        K = torch.transpose(torch.reshape(K, new_shape), 1, 2)
        V = torch.transpose(torch.reshape(V, new_shape), 1, 2)
        
        scores = torch.matmul(Q, torch.transpose(K, -1, -2))/np.sqrt(self.each_head_dim)
        
        if mask is not None:
            scores -= 10000.0 * (1.0 - mask)
            
        scores  = self.attn_dropout(self.softmax(scores))
        
        Z = torch.matmul(scores, V)
        Z = torch.flatten(torch.transpose(Z, 1, 2), start_dim=2)
        
        out = self.linear(Z)
        
        return out


# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForwardLayer(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=config.emb_dim, out_features=config.hidden_dim)
        self.fc2 = nn.Linear(in_features=config.hidden_dim, out_features=config.emb_dim)
        self.gelu = GELU()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=config.emb_dim, eps=config.eps)
        self.fc_dropout = nn.Dropout(p = config.fc_drop_rate)
        
    def forward(self, X):
        
        out = self.fc1(X)
        out = self.gelu(out)
        out = self.fc2(out)
        
        out = self.fc_dropout(out)
        
        out += X
        out = self.layer_norm(out)
        
        return out


class EncoderLayer(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttentionLayer(config)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.emb_dim, eps=config.eps)
        self.enc_drop = nn.Dropout(p = config.attn_drop_rate)
        
        self.feed_forward = FeedForwardLayer(config)
        
    def forward(self, embeddings, mask=None):
        
        out = self.self_attention(embeddings, mask)
        out = self.layer_norm(embeddings + self.enc_drop(out))
        out = self.feed_forward(out)
        
        return out


class BERTEncoder(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        self.enbedding_layer = EmbeddingLayer(config)
        
        self.encoders = nn.ModuleList([EncoderLayer(config) for i in range(config.num_encoders)])
        
    def forward(self, token_ip, sent_ip, pos_ip, mask=None):
        
        embeddings = self.enbedding_layer(token_ip, sent_ip, pos_ip)
        
        for encoder in self.encoders:
            
            embeddings = encoder(embeddings, mask)
            
        return embeddings



# https://github.com/dhlee347/pytorchic-bert/blob/master/checkpoint.py

""" Load a checkpoint file of pretrained transformer to a model in pytorch """

import numpy as np
import tensorflow as tf
import torch
#import ipdb
#from models import *

def load_param(checkpoint_file, conversion_table):
    """
    Load parameters in pytorch model from checkpoint file according to conversion_table
    checkpoint_file : pretrained checkpoint model file in tensorflow
    conversion_table : { pytorch tensor in a model : checkpoint variable name }
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

        # for weight(kernel), we should do transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        assert pyt_param.size() == tf_param.shape,             'Dim Mismatch: %s vs %s ; %s' %                 (tuple(pyt_param.size()), tf_param.shape, tf_param_name)
        
        # assign pytorch tensor from tensorflow param
        pyt_param.data = torch.from_numpy(tf_param)


def load_model(model, checkpoint_file):
    """ Load the pytorch model from checkpoint file """

    # Embedding layer
    e, p = model.enbedding_layer, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.token_embeddings.weight: p+"word_embeddings",
        e.positional_embeddings.weight: p+"position_embeddings",
        e.sentence_embeddings.weight: p+"token_type_embeddings",
        e.layer_norm.weight:       p+"LayerNorm/gamma",
        e.layer_norm.bias:        p+"LayerNorm/beta"
    })

    # Transformer blocks
    for i in range(len(model.encoders)):
        b, p = model.encoders[i], "bert/encoder/layer_%d/"%i
        load_param(checkpoint_file, {
            b.self_attention.queries.weight:   p+"attention/self/query/kernel",
            b.self_attention.queries.bias:     p+"attention/self/query/bias",
            b.self_attention.keys.weight:   p+"attention/self/key/kernel",
            b.self_attention.keys.bias:     p+"attention/self/key/bias",
            b.self_attention.values.weight:   p+"attention/self/value/kernel",
            b.self_attention.values.bias:     p+"attention/self/value/bias",
            b.self_attention.linear.weight:          p+"attention/output/dense/kernel",
            b.self_attention.linear.bias:            p+"attention/output/dense/bias",
            b.feed_forward.fc1.weight:      p+"intermediate/dense/kernel",
            b.feed_forward.fc1.bias:        p+"intermediate/dense/bias",
            b.feed_forward.fc2.weight:      p+"output/dense/kernel",
            b.feed_forward.fc2.bias:        p+"output/dense/bias",
            b.layer_norm.weight:          p+"attention/output/LayerNorm/gamma",
            b.layer_norm.bias:           p+"attention/output/LayerNorm/beta",
            b.feed_forward.layer_norm.weight:          p+"output/LayerNorm/gamma",
            b.feed_forward.layer_norm.bias:           p+"output/LayerNorm/beta",
        })

if __name__ == "__main__":

    config = Config()

    model = BERTEncoder(config)
    weights_path = "weights_uncased/bert_model.ckpt"
    load_model(model, weights_path)


    pos_ip = torch.unsqueeze(torch.arange(0, 512, dtype=torch.long), dim=0)
    token_ip = torch.randint(low=0, high=30000, size=(1, 512)).type(torch.LongTensor)
    sent_ip = torch.zeros(size=(1, 512), dtype=torch.long)
    sent_ip[:, 256:] = 1

    out = model(token_ip, sent_ip, pos_ip)


