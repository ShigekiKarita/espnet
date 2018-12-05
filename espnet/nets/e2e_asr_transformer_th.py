import logging
import math
import chainer
import torch
from torch import nn


class MultiSequential(torch.nn.Sequential):
    def forward(self, *args):
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    return MultiSequential(*[fn() for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask):
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, mask))
        nx = self.norm2(x)
        return x + self.dropout(self.feed_forward(nx)), mask


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, src_mask, x, tgt_mask):
        """
        :param torch.Tensor memory: encoded input features (batch, max_time_in, size)
        :param torch.Tensor src_mask: mask for memory (batch, max_time_in)
        :param torch.Tensor x: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        """
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, tgt_mask))
        nx = self.norm2(x)
        x = x + self.dropout(self.src_attn(nx, memory, memory, src_mask))
        nx = self.norm3(x)
        return memory, src_mask, x + self.dropout(self.feed_forward(nx)), tgt_mask


def subsequent_mask(size, device="cpu", dtype=torch.uint8):
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.triu(ret, out=ret).unsqueeze(0)


import numpy
MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1)
        :param torch.nn.Dropout dropout:
        """
        # logging.warning("q, k, v, m = {}, {}, {}, {}".format(query.shape, key.shape, value.shape, mask.shape))
        mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        # (batch, head, time1/2, d_k)
        q = self.linear_q(query).view(n_batch, self.h, -1, self.d_k)
        k = self.linear_k(key).view(n_batch, self.h, -1, self.d_k)
        v = self.linear_v(value).view(n_batch, self.h, -1, self.d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # logging.warning("{} vs {}".format(scores.shape, mask.shape))
        scores = scores.masked_fill(mask == 0, MIN_VALUE)
        self.attn = torch.softmax(scores, dim = -1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)


class Encoder(torch.nn.Module):
    def __init__(self, idim, args):
        super(Encoder, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, args.adim),
            torch.nn.Dropout(args.dropout_rate),
            torch.nn.ReLU()
        )
        self.encoders = repeat(
            args.elayers,
            lambda : EncoderLayer(
                args.adim,
                MultiHeadedAttention(args.aheads, args.adim, args.dropout_rate),
                PositionwiseFeedForward(args.adim, args.eunits, args.dropout_rate),
                args.dropout_rate
            )
        )
        self.norm = LayerNorm(args.adim)

    def forward(self, x, mask):
        x = self.input_layer(x)
        return self.encoders(x, mask)


class Decoder(torch.nn.Module):
    def __init__(self, odim, args):
        super(Decoder, self).__init__()
        self.embed = torch.nn.Sequential(
            Embeddings(args.adim, odim),
            PositionalEncoding(args.adim, args.dropout_rate)
        )
        self.decoders = repeat(
            args.dlayers,
            lambda : DecoderLayer(
                args.adim,
                MultiHeadedAttention(args.aheads, args.adim, args.dropout_rate),
                MultiHeadedAttention(args.aheads, args.adim, args.dropout_rate),
                PositionwiseFeedForward(args.adim, args.dunits, args.dropout_rate),
                args.dropout_rate
            )
        )
        self.output_norm = LayerNorm(args.adim)
        self.output_layer = torch.nn.Linear(args.adim, odim)

    def forward(self, memory, memory_mask, tgt, tgt_mask):
        x = self.embed(tgt)
        memory, memory_mask, x, tgt_mask = self.decoders(memory, memory_mask, x, tgt_mask)
        x = self.output_layer(self.output_norm(x))
        return x, tgt_mask


class E2E(torch.nn.Module):
    def __init__(self, idim, odim, args):
        logging.info("initializing transformer E2E")
        super(E2E, self).__init__()
        self.encoder = Encoder(idim, args)
        self.decoder = Decoder(odim, args)
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = -1
        self.subsample = [0]
        # self.char_list = args.char_list
        # self.verbose = args.verbose
        self.reset_parameters(args)

    def reset_parameters(self, args):
        if args.ninit == "none":
            return
        for p in self.parameters():
            if p.dim() == 2:
                if args.ninit == "chainer":
                    stdv = 1. / math.sqrt(p.data.size(1))
                    p.data.normal_(0, stdv)
                elif args.ninit == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.ninit == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.ninit == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.ninit == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.ninit)

        # zero bias
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        self.decoder.embed[0].lut.weight.data.normal_(0, 1)



    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        from espnet.nets.e2e_asr_th import make_pad_mask, pad_list, th_accuracy
        # 1. encoder
        src_mask = (~make_pad_mask(ilens)).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        eos = ys_pad.new([self.eos])
        sos = ys_pad.new([self.sos])
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_mask = ys_in_pad != self.ignore_id
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device)
        ys_mask = ys_mask.unsqueeze(-2) & m
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        pred_pad, pred_mask = self.decoder(hs_pad, hs_mask, ys_in_pad, ys_mask)
        # logging.warning("{} vs {}".format(pred_pad.shape, ys_out_pad.shape))
        loss_att = torch.nn.functional.cross_entropy(
            pred_pad.view(-1, self.odim),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            size_average=True)
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                          ignore_label=self.ignore_id)
        # TODO(karita)
        loss_ctc = None
        cer, wer = 0.0, 0.0
        return loss_ctc, loss_att, acc, cer, wer


    def calculate_all_attentions(self, hs_pad, hlen, ys_pad):
        pass
