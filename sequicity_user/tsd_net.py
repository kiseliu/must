import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from sequicity_user.config import global_config as cfg
import copy, random, time, logging

from torch.distributions import Categorical
from sequicity_user.reader import pad_sequences
import pdb


def cuda_(var):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    if type(v) is float:
        return v == float('nan')
    return np.isnan(np.sum(v.data.cpu().numpy()))


def get_sparse_input_aug(x_input_np):
    """
    sparse input of
    :param x_input_np: [T,B]
    :return: Numpy array: [B,T,aug_V]
    """
    ignore_index = [0]
    unk = 2
    result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
                      dtype=np.float32)
    result.fill(1e-10)
    for t in range(x_input_np.shape[0]):
        for b in range(x_input_np.shape[1]):
            w = x_input_np[t][b]
            if w not in ignore_index:
                if w != unk:
                    result[t][b][x_input_np[t][b]] = 1.0
                else:
                    result[t][b][cfg.vocab_size + t] = 1.0
    result_np = result.transpose((1, 0, 2))
    result = torch.from_numpy(result_np).float()
    return result


def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, mask=False, inp_seqs=None, stop_tok=None, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        if True or not mask:
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        else:
            mask_idx = []
            # inp_seqs: ndarray of [T,B]
            # inp_seqs = inp_seqs.cpu().numpy()
            for b in range(inp_seqs.shape[1]):
                for t in range(inp_seqs.shape[0] + 1):
                    if t == inp_seqs.shape[0] or inp_seqs[t, b] in stop_tok:
                        mask_idx.append(t)
                        break
            mask = []
            for mask_len in mask_idx:
                mask.append([1.] * mask_len + [0.] * (inp_seqs.shape[0] - mask_len))
            mask = cuda_(Variable(torch.FloatTensor(mask)))  # [B,T]
            attn_energies = attn_energies * mask.unsqueeze(1)
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]

        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context.transpose(0, 1)  # [1,B,H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class SimpleDynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        init_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None, enc_rand=False):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden: num_layers * num_directions, batch, hidden_size
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        if enc_rand is True:
            if cfg.cuda:
                hidden = torch.rand(cfg.layer_num*2, len(input_lens), self.hidden_size).cuda()
            else:
                hidden = torch.rand(cfg.layer_num*2, len(input_lens), self.hidden_size)
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded

class BSpanDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        if cfg.use_positional_embedding:
            self.positional_embedding = nn.Embedding(cfg.max_ts + 1, embed_size)
            init_pos_emb = self.position_encoding_init(cfg.max_ts + 1, embed_size)
            self.positional_embedding.weight.data = init_pos_emb
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, dropout=dropout_rate)
        self.proj = nn.Linear(hidden_size * 2, vocab_size)

        self.attn_u = Attn(hidden_size)
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

        self.inp_dropout = nn.Dropout(self.dropout_rate)

        init_gru(self.gru)
        self.vocab = vocab

    def position_encoding_init(self, n_position, d_pos_vec):
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
                                 if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, u_enc_out, z_tm1, last_hidden, u_input_np, pv_z_enc_out, prev_z_input_np, u_emb, pv_z_emb,
                position):

        sparse_u_input = cuda_(Variable(get_sparse_input_aug(u_input_np), requires_grad=False))

        if pv_z_enc_out is not None:
            context = self.attn_u(last_hidden, torch.cat([pv_z_enc_out, u_enc_out], dim=0), mask=True,
                                  inp_seqs=np.concatenate([prev_z_input_np, u_input_np], 0),
                                  stop_tok=[self.vocab.encode('EOS_M')])
        else:
            context = self.attn_u(last_hidden, u_enc_out, mask=True, inp_seqs=u_input_np,
                                  stop_tok=[self.vocab.encode('EOS_M')])
        embed_z = self.emb(z_tm1)
        # embed_z = self.inp_dropout(embed_z)

        if cfg.use_positional_embedding:  # defaulty not used
            position_label = [position] * u_enc_out.size(1)  # [B]
            position_label = cuda_(Variable(torch.LongTensor(position_label))).view(1, -1)  # [1,B]
            pos_emb = self.positional_embedding(position_label)
            embed_z = embed_z + pos_emb

        gru_in = torch.cat([embed_z, context], 2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        # gru_out = self.inp_dropout(gru_out)
        gen_score = self.proj(torch.cat([gru_out, context], 2)).squeeze(0)
        # gen_score = self.inp_dropout(gen_score)
        u_copy_score = torch.tanh(self.proj_copy1(u_enc_out.transpose(0, 1)))  # [B,T,H]
        # stable version of copynet
        u_copy_score = torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        # u_copy_score = u_copy_score.cpu()
        u_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0]
        u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
        u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(
            1) + u_copy_score_max  # [B,V]
        u_copy_score = cuda_(u_copy_score)
        if pv_z_enc_out is None:
            # u_copy_score = self.inp_dropout(u_copy_score)
            scores = F.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)
            gen_score, u_copy_score = scores[:, :cfg.vocab_size], \
                                      scores[:, cfg.vocab_size:]
            proba = gen_score + u_copy_score[:, :cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, u_copy_score[:, cfg.vocab_size:]], 1)
        else:
            sparse_pv_z_input = cuda_(Variable(get_sparse_input_aug(prev_z_input_np), requires_grad=False))
            pv_z_copy_score = torch.tanh(self.proj_copy2(pv_z_enc_out.transpose(0, 1)))  # [B,T,H]
            pv_z_copy_score = torch.matmul(pv_z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            # pv_z_copy_score = pv_z_copy_score.cpu()
            pv_z_copy_score_max = torch.max(pv_z_copy_score, dim=1, keepdim=True)[0]
            pv_z_copy_score = torch.exp(pv_z_copy_score - pv_z_copy_score_max)  # [B,T]
            pv_z_copy_score = torch.log(torch.bmm(pv_z_copy_score.unsqueeze(1), sparse_pv_z_input)).squeeze(
                1) + pv_z_copy_score_max  # [B,V]
            pv_z_copy_score = cuda_(pv_z_copy_score)
            scores = F.softmax(torch.cat([gen_score, u_copy_score, pv_z_copy_score], dim=1), dim=1)
            gen_score, u_copy_score, pv_z_copy_score = scores[:, :cfg.vocab_size], \
                                                       scores[:,
                                                       cfg.vocab_size:2 * cfg.vocab_size + u_input_np.shape[0]], \
                                                       scores[:, 2 * cfg.vocab_size + u_input_np.shape[0]:]
            proba = gen_score + u_copy_score[:, :cfg.vocab_size] + pv_z_copy_score[:, :cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, pv_z_copy_score[:, cfg.vocab_size:], u_copy_score[:, cfg.vocab_size:]], 1)
        return gru_out, last_hidden, proba

class ResponseDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate, gru, proj, emb, vocab):
        super().__init__()
        self.emb = emb
        self.attn_z = Attn(hidden_size)
        self.attn_u = Attn(hidden_size)
        self.attn_g = Attn(hidden_size)
        self.gru = gru
        init_gru(self.gru)
        self.proj = proj
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy3 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

        self.vocab = vocab

    def get_sparse_selective_input(self, x_input_np):
        result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
                          dtype=np.float32)
        result.fill(1e-10)
        reqs = ['address', 'phone', 'postcode', 'pricerange', 'area']
        for t in range(x_input_np.shape[0] - 1):
            for b in range(x_input_np.shape[1]):
                w = x_input_np[t][b]
                word = self.vocab.decode(w)
                if word in reqs:
                    slot = self.vocab.encode(word + '_SLOT')
                    result[t + 1][b][slot] = 1.0
                else:
                    if w == 2 or w >= cfg.vocab_size:
                        result[t + 1][b][cfg.vocab_size + t] = 5.0
                    else:
                        result[t + 1][b][w] = 1.0
        result_np = result.transpose((1, 0, 2))
        result = torch.from_numpy(result_np).float()
        return result

    def forward(self, z_enc_out, u_enc_out, u_input_np, m_t_input, degree_input, last_hidden, z_input_np, g_enc_out=None, last_hidden_g=None,**kwargs):
        # # adding goal for user utterance decoding
        g_input_np = kwargs.get('g_input_np', None)
        if g_input_np is None:
            sparse_z_input = cuda_(Variable(self.get_sparse_selective_input(z_input_np), requires_grad=False))

            m_embed = self.emb(m_t_input)
            z_context = self.attn_z(last_hidden, z_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_Z2')],
                                    inp_seqs=z_input_np)
            u_context = self.attn_u(last_hidden, u_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_M')],
                                    inp_seqs=u_input_np)
            gru_in = torch.cat([m_embed, u_context, z_context, degree_input.unsqueeze(0)], dim=2)
            gru_out, last_hidden = self.gru(gru_in, last_hidden)
            gen_score = self.proj(torch.cat([z_context, u_context, gru_out], 2)).squeeze(0)
            z_copy_score = torch.tanh(self.proj_copy2(z_enc_out.transpose(0, 1)))
            z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            # z_copy_score = z_copy_score.cpu()
            z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0]
            z_copy_score = torch.exp(z_copy_score - z_copy_score_max)  # [B,T]
            z_copy_score = torch.log(torch.bmm(z_copy_score.unsqueeze(1), sparse_z_input)).squeeze(
                1) + z_copy_score_max  # [B,V]
            z_copy_score = cuda_(z_copy_score)

            scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1)
            gen_score, z_copy_score = scores[:, :cfg.vocab_size], \
                                      scores[:, cfg.vocab_size:]
            proba = gen_score + z_copy_score[:, :cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, z_copy_score[:, cfg.vocab_size:]], 1)
            return proba, last_hidden, gru_out
        else:
            sparse_z_input = cuda_(Variable(self.get_sparse_selective_input(z_input_np), requires_grad=False)) # [B,T,V+T]

            # pdb.set_trace()
            m_embed = self.emb(m_t_input)
            z_context = self.attn_z(last_hidden, z_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_Z2')],
                                    inp_seqs=z_input_np)
            u_context = self.attn_u(last_hidden, u_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_M')],
                                    inp_seqs=u_input_np)
            g_context = self.attn_g(last_hidden_g, g_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_Z0')],
                                    inp_seqs=g_input_np)
            gru_in = torch.cat([m_embed, u_context, z_context, g_context, degree_input.unsqueeze(0)], dim=2)
            gru_out, last_hidden = self.gru(gru_in, last_hidden)
            gen_score = self.proj(torch.cat([z_context, u_context, g_context, gru_out], 2)).squeeze(0)

            z_copy_score = torch.tanh(self.proj_copy2(z_enc_out.transpose(0, 1))) # [B,T,H] #z_enc_out, [T,B,H]
            z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2) # [B,T]
            # z_copy_score = z_copy_score.cpu()
            z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0]
            z_copy_score = torch.exp(z_copy_score - z_copy_score_max)  # [B,T]
            z_copy_score = torch.log(torch.bmm(z_copy_score.unsqueeze(1), sparse_z_input)).squeeze(
                1) + z_copy_score_max  # [B,V]
            z_copy_score = cuda_(z_copy_score)

            goal = cuda_(Variable(torch.from_numpy(g_input_np).long()))
            sparse_g_input = cuda_(Variable(self.get_sparse_selective_input(goal), requires_grad=False))
            g_enc_out = self.emb(goal)
            g_copy_score = torch.tanh(self.proj_copy3(g_enc_out.transpose(0, 1)))
            g_copy_score = torch.matmul(g_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
            # g_copy_score = g_copy_score.cpu()
            g_copy_score_max = torch.max(g_copy_score, dim=1, keepdim=True)[0]
            g_copy_score = torch.exp(g_copy_score - g_copy_score_max)  # [B,T]
            g_copy_score = torch.log(torch.bmm(g_copy_score.unsqueeze(1), sparse_g_input)).squeeze(
                1) + g_copy_score_max  # [B,V]
            g_copy_score = cuda_(g_copy_score)

            scores = F.softmax(torch.cat([gen_score, z_copy_score, g_copy_score], dim=1), dim=1)
            gen_score, z_copy_score, g_copy_score = scores[:, :cfg.vocab_size], \
                                                    scores[:, cfg.vocab_size: 2 * cfg.vocab_size + z_input_np.shape[0]], \
                                                    scores[:, 2 * cfg.vocab_size + z_input_np.shape[0]:]
            proba = gen_score + z_copy_score[:, :cfg.vocab_size] + g_copy_score[:, :cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, z_copy_score[:, cfg.vocab_size:], g_copy_score[:, cfg.vocab_size:]], 1)
            return proba, last_hidden, gru_out

class ResponseActClassifier(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.gru_u = nn.GRU(hidden_size , hidden_size, dropout=dropout_rate)
        self.gru_z = nn.GRU(hidden_size , hidden_size, dropout=dropout_rate)
        self.gru_g = nn.GRU(hidden_size , hidden_size, dropout=dropout_rate)
        init_gru(self.gru_u)
        init_gru(self.gru_z)
        init_gru(self.gru_g)
        self.linear_1 = nn.Linear(3 * hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 7)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, z_enc_out, last_hidden_z, u_enc_out, last_hidden_u, g_enc_out, last_hidden_g,  **kwargs):
        u_context, last_hidden_u = self.gru_u(u_enc_out, last_hidden_u)
        z_context, last_hidden_z = self.gru_z(z_enc_out, last_hidden_z)
        if g_enc_out is None:
            hidden = torch.tanh(self.linear_1(torch.cat([last_hidden_u, last_hidden_z], dim=2)))
            output = self.softmax(self.linear_2(hidden))
            return output
        else:
            g_context, last_hidden_g = self.gru_g(g_enc_out, last_hidden_g)
            hidden = torch.tanh(self.linear_1(torch.cat([last_hidden_u, last_hidden_z, last_hidden_g], dim=2)))
            output = self.softmax(self.linear_2(hidden))#.squeeze(0)
            return output

class TSD(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, layer_num, dropout_rate, z_length,
                 max_ts, beam_search=False, teacher_force=100, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.reader = kwargs['reader']
        self.emb = nn.Embedding(vocab_size, embed_size)
        if 'usr' in cfg.dataset:
            self.dec_gru = nn.GRU(degree_size + embed_size + hidden_size * 3, hidden_size, dropout=dropout_rate)
            self.proj = nn.Linear(hidden_size * 4, vocab_size)
        else:
            self.dec_gru = nn.GRU(degree_size + embed_size + hidden_size * 2, hidden_size, dropout=dropout_rate)
            self.proj = nn.Linear(hidden_size * 3, vocab_size)
        
        self.u_encoder = SimpleDynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.z_decoder = BSpanDecoder(embed_size, hidden_size, vocab_size, dropout_rate, self.vocab)
        self.m_decoder = ResponseDecoder(embed_size, hidden_size, vocab_size, degree_size, dropout_rate,
                                         self.dec_gru, self.proj, self.emb, self.vocab)
        self.act_classifier = ResponseActClassifier(hidden_size, dropout_rate)
        self.embed_size = embed_size

        self.z_length = z_length
        self.max_ts = max_ts
        self.beam_search = beam_search
        self.teacher_force = teacher_force

        self.pr_loss = nn.NLLLoss(ignore_index=0)
        self.dec_loss = nn.NLLLoss(ignore_index=0)
        self.act_loss = nn.NLLLoss()

        self.saved_log_policy = []

        if self.beam_search:
            self.beam_size = kwargs['beam_size']
            self.eos_token_idx = kwargs['eos_token_idx']

    def forward(self, u_input, u_input_np, m_input, m_input_np, z_input, u_len, m_len, turn_states,
                degree_input, mode, **kwargs):
        if mode == 'train' or mode == 'valid':
            pz_proba, pm_dec_proba, turn_states = \
                self.forward_turn(u_input, u_len, m_input=m_input, m_len=m_len, z_input=z_input, mode='train',
                                  turn_states=turn_states, degree_input=degree_input, u_input_np=u_input_np,
                                  m_input_np=m_input_np, **kwargs)
            loss, pr_loss, m_loss = self.supervised_loss(torch.log(pz_proba), torch.log(pm_dec_proba),
                                                         z_input, m_input)
            return loss, pr_loss, m_loss, turn_states

        elif mode == 'test':
            m_output_index, pz_index, turn_states = self.forward_turn(u_input, u_len=u_len, mode='test',
                                                                      turn_states=turn_states,
                                                                      degree_input=degree_input,
                                                                      u_input_np=u_input_np, m_input_np=m_input_np,
                                                                      **kwargs
                                                                      )
            return m_output_index, pz_index, turn_states
        elif mode == 'rl':
            loss = self.forward_turn(u_input, u_len=u_len, is_train=False, mode='rl',
                                     turn_states=turn_states,
                                     degree_input=degree_input,
                                     u_input_np=u_input_np, m_input_np=m_input_np,
                                     **kwargs
                                     )
            return loss

    def forward_turn(self, u_input, u_len, turn_states, mode, degree_input, u_input_np, m_input_np=None,
                     m_input=None, m_len=None, z_input=None, **kwargs):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_input_np:
        :param m_input_np:
        :param u_len:
        :param turn_states:
        :param is_train:
        :param u_input: [T,B]
        :param m_input: [T,B]
        :param z_input: [T,B]
        :return:
        """
        prev_z_input = kwargs.get('prev_z_input', None)
        prev_z_input_np = kwargs.get('prev_z_input_np', None)
        prev_z_len = kwargs.get('prev_z_len', None)

        g_input = kwargs.get('g_input', None)
        g_input_np = kwargs.get('g_input_np', None)
        g_input_len = kwargs.get('g_input_len', None)
        g_enc_out = None
        last_hidden_g = None

        pv_z_emb = None
        batch_size = u_input.size(1)
        pv_z_enc_out = None
        model_name = kwargs.get('model', None)
        goal = kwargs.get('goal', None)

        if mode == 'train':
            enc_rand = False
        else:
            enc_rand = True

        if prev_z_input is not None:
            pv_z_enc_out, _, pv_z_emb = self.u_encoder(prev_z_input, prev_z_len, enc_rand=enc_rand)

        if g_input is not None:
            g_enc_out, g_enc_hidden, g_emb = self.u_encoder(g_input, g_input_len, enc_rand=enc_rand)
            last_hidden_g = g_enc_hidden[:-1]

        u_enc_out, u_enc_hidden, u_emb = self.u_encoder(u_input, u_len, enc_rand=enc_rand)
        last_hidden = u_enc_hidden[:-1]
        z_tm1 = cuda_(Variable(torch.ones(1, batch_size).long() * 3))  # GO_2 token
        m_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token
        if mode == 'train':
            pz_dec_outs = []
            pz_proba = []
            z_length = z_input.size(0) if z_input is not None else self.z_length  # GO token
            hiddens = [None] * batch_size
            for t in range(z_length):
                pz_dec_out, last_hidden, proba = \
                    self.z_decoder(u_enc_out=u_enc_out, u_input_np=u_input_np,
                                   z_tm1=z_tm1, last_hidden=last_hidden,
                                   pv_z_enc_out=pv_z_enc_out, prev_z_input_np=prev_z_input_np,
                                   u_emb=u_emb, pv_z_emb=pv_z_emb, position=t)
                pz_proba.append(proba)
                pz_dec_outs.append(pz_dec_out)
                z_np = z_tm1.view(-1).cpu().data.numpy()
                for i in range(batch_size):
                    if z_np[i] == self.vocab.encode('EOS_Z2'):
                        hiddens[i] = last_hidden[:, i, :]
                z_tm1 = z_input[t].view(1, -1)
            for i in range(batch_size):
                if hiddens[i] is None:
                    hiddens[i] = last_hidden[:, i, :]
            last_hidden = torch.stack(hiddens, dim=1)

            z_input_np = z_input.cpu().data.numpy()

            pz_dec_outs = torch.cat(pz_dec_outs, dim=0)  # [Tz,B,H]
            pz_proba = torch.stack(pz_proba, dim=0)
            # P(m|z,u)
            # pdb.set_trace()
            if cfg.dataset == 'usr_act':
                # # _enc_out [T,B,H]
                # # last_hidden [1,B,H]
                act_proba = self.act_classifier(z_enc_out=pz_dec_outs, last_hidden_z=last_hidden, \
                                                u_enc_out=u_enc_out, last_hidden_u=u_enc_hidden[:-1], \
                                                g_enc_out=g_enc_out, last_hidden_g=last_hidden_g, **kwargs)
                return pz_proba, act_proba , None
            else:
                pm_dec_proba, m_dec_outs = [], []
                m_length = m_input.size(0)  # Tm
                # last_hidden = u_enc_hidden[:-1]
                for t in range(m_length):
                    teacher_forcing = toss_(self.teacher_force)
                    proba, last_hidden, dec_out = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1,
                                                                 degree_input, last_hidden, z_input_np,
                                                                 g_enc_out=g_enc_out, last_hidden_g=last_hidden_g, **kwargs)
                    if teacher_forcing:
                        m_tm1 = m_input[t].view(1, -1)
                    else:
                        _, m_tm1 = torch.topk(proba, 1)
                        m_tm1 = m_tm1.view(1, -1)
                    pm_dec_proba.append(proba)
                    m_dec_outs.append(dec_out)

                pm_dec_proba = torch.stack(pm_dec_proba, dim=0)  # [T,B,V]
                return pz_proba, pm_dec_proba, None
        else:
            pz_dec_outs, bspan_index, last_hidden = self.bspan_decoder(u_enc_out, z_tm1, last_hidden, u_input_np,
                                                                       pv_z_enc_out=pv_z_enc_out,
                                                                       prev_z_input_np=prev_z_input_np,
                                                                       u_emb=u_emb, pv_z_emb=pv_z_emb)
            pz_dec_outs = torch.cat(pz_dec_outs, dim=0)
            degree_input = self.reader.db_degree_handler(bspan_index, kwargs['dial_id'])
            degree_input = cuda_(Variable(torch.from_numpy(degree_input).float()))
            if cfg.dataset == 'usr_act' and mode == 'test':
                act_proba = self.act_classifier(z_enc_out=pz_dec_outs, last_hidden_z=last_hidden, \
                                                u_enc_out=u_enc_out, last_hidden_u=u_enc_hidden[:-1], \
                                                g_enc_out=g_enc_out, last_hidden_g=last_hidden_g, **kwargs)
                # pdb.set_trace()
                _, m_output_index = torch.topk(act_proba, 1) #[T,B,1]
                return m_output_index, bspan_index, None

            elif mode == 'test':
                if not self.beam_search:
                    m_output_index = self.greedy_decode(pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden,
                                                        degree_input, bspan_index, 
                                                        g_enc_out=g_enc_out, last_hidden_g=last_hidden_g,**kwargs)

                else:
                    m_output_index = self.beam_search_decode(pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden,
                                                             degree_input, bspan_index)

                return m_output_index, bspan_index, None
            elif mode == 'rl':
                return self.sampling_decode(pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden,
                                            degree_input, bspan_index)

    def bspan_decoder(self, u_enc_out, z_tm1, last_hidden, u_input_np, pv_z_enc_out, prev_z_input_np, u_emb, pv_z_emb):
        pz_dec_outs = []
        pz_proba = []
        decoded = []
        batch_size = u_enc_out.size(1)
        hiddens = [None] * batch_size
        for t in range(cfg.z_length):
            pz_dec_out, last_hidden, proba = \
                self.z_decoder(u_enc_out=u_enc_out, u_input_np=u_input_np,
                               z_tm1=z_tm1, last_hidden=last_hidden, pv_z_enc_out=pv_z_enc_out,
                               prev_z_input_np=prev_z_input_np, u_emb=u_emb, pv_z_emb=pv_z_emb, position=t)
            pz_proba.append(proba)
            pz_dec_outs.append(pz_dec_out)
            z_proba, z_index = torch.topk(proba, 1)  # [B,1]
            z_index = z_index.data.view(-1)

            #####################################################
            if prev_z_input_np is None:
                tmp = u_input_np   # [,B]
            else:
                # pdb.set_trace()
                tmp = np.concatenate((u_input_np, prev_z_input_np), axis=0)
                
            for i in range(z_index.size(0)):
                if z_index[i] >= cfg.vocab_size:
                    z_index[i] = torch.tensor(int(tmp[z_index[i]-cfg.vocab_size, i]))
            del tmp
            decoded.append(z_index.clone())
            #####################################################

            for i in range(z_index.size(0)):
                if z_index[i] >= cfg.vocab_size:
                    z_index[i] = 2  # unk
            z_np = z_tm1.view(-1).cpu().data.numpy()
            for i in range(batch_size):
                if z_np[i] == self.vocab.encode('EOS_Z2'):
                    hiddens[i] = last_hidden[:, i, :]
            z_tm1 = cuda_(Variable(z_index).view(1, -1))
        for i in range(batch_size):
            if hiddens[i] is None:
                hiddens[i] = last_hidden[:, i, :]
        last_hidden = torch.stack(hiddens, dim=1)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        decoded = [list(_) for _ in decoded]
        return pz_dec_outs, decoded, last_hidden

    def greedy_decode(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index, g_enc_out,last_hidden_g,  **kwargs):
        decoded = []
        bspan_index_np = pad_sequences(bspan_index).transpose((1, 0))
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1,
                                                   degree_input, last_hidden, bspan_index_np,
                                                   g_enc_out=g_enc_out, last_hidden_g=last_hidden_g, **kwargs)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= cfg.vocab_size:
                    mt_index[i] = 2  # unk
            m_tm1 = cuda_(Variable(mt_index).view(1, -1))
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def beam_search_decode_single(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input,
                                  bspan_index):
        eos_token_id = self.vocab.encode(cfg.eos_m_token)
        batch_size = pz_dec_outs.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def beam_result_valid(decoded_t, bspan_index):
            decoded_t = [_.view(-1).data[0] for _ in decoded_t]
            req_slots = self.get_req_slots(bspan_index)
            decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
            for req in req_slots:
                if req not in decoded_sentence:
                    return False
            return True

        def score_bonus(state, decoded, bspan_index):
            bonus = cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [m_tm1], 0))
        bspan_index_np = np.array(bspan_index).reshape(-1, 1)
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, m_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1, degree_input,
                                                       last_hidden, bspan_index_np)

                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].data[0], t) + score_bonus(state,
                                                                                                mt_index[0][new_k].data[
                                                                                                    0], bspan_index)
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.data[0] >= cfg.vocab_size:
                        decoded_t.data[0] = 2  # unk
                    if self.vocab.decode(decoded_t.data[0]) == cfg.eos_m_token:
                        if beam_result_valid(state.decoded, bspan_index):
                            finished.append(state)
                            dead_k += 1
                        else:
                            failed.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).data[0] for _ in decoded_t]
        decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
        print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index):
        vars = torch.split(pz_dec_outs, 1, dim=1), torch.split(u_enc_out, 1, dim=1), torch.split(
            m_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1), torch.split(degree_input, 1, dim=0)
        decoded = []
        for i, (pz_dec_out_s, u_enc_out_s, m_tm1_s, last_hidden_s, degree_input_s) in enumerate(zip(*vars)):
            decoded_s = self.beam_search_decode_single(pz_dec_out_s, u_enc_out_s, m_tm1_s,
                                                       u_input_np[:, i].reshape((-1, 1)),
                                                       last_hidden_s, degree_input_s, bspan_index[i])
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def supervised_loss(self, pz_proba, pm_dec_proba, z_input, m_input):
        pz_proba, pm_dec_proba = pz_proba[:, :, :cfg.vocab_size].contiguous(), pm_dec_proba[:, :,
                                                                               :cfg.vocab_size].contiguous()
        pr_loss = self.pr_loss(pz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        if cfg.dataset == 'usr_act':
            m_loss = self.act_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input[0].view(-1))
        else:
            m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))

        loss = pr_loss + m_loss
        return loss, pr_loss, m_loss

    def self_adjust(self, epoch):
        pass

    # REINFORCEMENT fine-tuning with MC

    def possible_reqs(self):
        if cfg.dataset == 'camrest':
            return ['address', 'phone', 'postcode', 'pricerange', 'area']
        elif cfg.dataset == 'kvret':
            req_by_intent = {
                'weather': ['weather_attribute'],
                'navigate': ['poi', 'traffic_info', 'address', 'distance'],
                'schedule': ['event', 'date', 'time', 'party', 'agenda', 'room']
            }
            reqs = []
            for value in req_by_intent.values():
                reqs.extend(value)
            return reqs
        else:
            raise ValueError('unknown dataset')

    def get_req_slots(self, bspan_index):
        reqs = self.possible_reqs()
        reqs = set(self.vocab.sentence_decode(bspan_index).split()).intersection(reqs)
        return [_ + '_SLOT' for _ in reqs]

    def reward(self, m_tm1, decoded, bspan_index):
        """
        The setting of the reward function is heuristic. It can be better optimized.
        :param m_tm1:
        :param decoded:
        :param bspan_index:
        :return:
        """
        req_slots = self.get_req_slots(bspan_index)

        m_tm1 = self.vocab.decode(m_tm1[0])
        finished = m_tm1 == 'EOS_M'
        decoded = [_.view(-1)[0] for _ in decoded]
        decoded_sentence = self.vocab.sentence_decode(decoded, cfg.eos_m_token).split()
        reward = -0.01 if cfg.dataset == 'camrest' else 0
        '''
        if not finished:
            if m_tm1 in req_slots:
                if decoded_sentence and m_tm1 not in decoded_sentence[:-1]:
                    reward = 1.0
        '''
        # some modification for reward function.
        if m_tm1 in req_slots:
            if decoded_sentence and m_tm1 not in decoded_sentence[:-1]:
                reward += 1.0
            else:
                reward -= 1.0  if cfg.dataset == 'camrest' else 0 # repeat
        return reward, finished

    def sampling_decode(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index):
        vars = torch.split(pz_dec_outs, 1, dim=1), torch.split(u_enc_out, 1, dim=1), torch.split(
            m_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1), torch.split(degree_input, 1, dim=0)
        batch_loss = []

        sample_num = 1

        for i, (pz_dec_out_s, u_enc_out_s, m_tm1_s, last_hidden_s, degree_input_s) in enumerate(zip(*vars)):
            if not self.get_req_slots(bspan_index[i]):
                continue
            for j in range(sample_num):
                loss = self.sampling_decode_single(pz_dec_out_s, u_enc_out_s, m_tm1_s,
                                                   u_input_np[:, i].reshape((-1, 1)),
                                                   last_hidden_s, degree_input_s, bspan_index[i])
                batch_loss.append(loss)
        if not batch_loss:
            return None
        else:
            return sum(batch_loss) / len(batch_loss)

    def sampling_decode_single(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index):
        decoded = []
        reward_sum = 0
        log_probs = []
        rewards = []
        bspan_index_np = np.array(bspan_index).reshape(-1, 1)
        for t in range(self.max_ts):
            # reward
            reward, finished = self.reward(m_tm1.data.view(-1), decoded, bspan_index)
            reward_sum += reward
            rewards.append(reward)
            if t == self.max_ts - 1:
                finished = True
            if finished:
                loss = self.finish_episode(log_probs, rewards)
                return loss
            # action
            proba, last_hidden, _ = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1,
                                                   degree_input, last_hidden, bspan_index_np)
            proba = proba.squeeze(0)  # [B,V]
            dis = Categorical(proba)
            action = dis.sample()
            log_probs.append(dis.log_prob(action))
            mt_index = action.data.view(-1)
            decoded.append(mt_index.clone())

            for i in range(mt_index.size(0)):
                if mt_index[i] >= cfg.vocab_size:
                    mt_index[i] = 2  # unk

            m_tm1 = cuda_(Variable(mt_index).view(1, -1))

    def finish_episode(self, log_probas, saved_rewards):
        R = 0
        policy_loss = []
        rewards = []
        for r in saved_rewards:
            R = r + 0.8 * R
            rewards.insert(0, R)

        rewards = torch.Tensor(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        for log_prob, reward in zip(log_probas, rewards):
            policy_loss.append(-log_prob * reward)
        l = len(policy_loss)
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss / l
