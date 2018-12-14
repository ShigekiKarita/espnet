import logging
import math
import chainer
import torch
from torch import nn
from torch.nn import LayerNorm

from espnet.asr import asr_utils
from espnet.nets.beam_search import BeamSearch, ScoringBase
from espnet.nets.e2e_asr_th import make_pad_mask, pad_list, th_accuracy


# TODO make this serializable
# from opennmt-py
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        ret = dict()
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model, d_model, warmup, factor):
    base = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, factor, warmup, base)


class MultiSequential(torch.nn.Sequential):
    def forward(self, *args):
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """repeat module N times
    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn() for _ in range(N)])


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
        """Compute encoded features
        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        """
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

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features
        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        x = tgt
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, tgt_mask))
        nx = self.norm2(x)
        x = x + self.dropout(self.src_attn(nx, memory, memory, memory_mask))
        nx = self.norm3(x)
        return x + self.dropout(self.feed_forward(nx)), tgt_mask, memory, memory_mask


def subsequent_mask(size, device="cpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)
    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


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
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)  # (batch, head, time1, d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)    # (batch, head, time2, d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)  # (batch, head, time2, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        mask = mask.unsqueeze(1)
        logging.debug("score {}, mask {}".format(scores.shape, mask.shape))
        scores = scores.masked_fill(mask == 0, MIN_VALUE)
        self.attn = torch.softmax(scores, dim = -1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x) # (batch, time1, d_model)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
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
        self.xscale = math.sqrt(d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * self.xscale + self.pe[:, :x.size(1)]
            return self.dropout(x)


class Conv2dLayerNorm(torch.nn.Module):
    def __init__(self, nin, nout, kernel, stride, **kwargs):
        super(Conv2dLayerNorm, self).__init__()
        self.conv = torch.nn.Conv2d(nin, nout, kernel, stride, **kwargs)
        self.norm = LayerNorm(nout)

    def forward(self, x):
        x = self.conv(x)  # (b, c, w, h)
        return self.norm(x.transpose(1, 3)).transpose(3, 1)


class Conv2dSubsampling(torch.nn.Module):
    def __init__(self, dim, dropout):
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            # Conv2dLayerNorm(1, dim, 3, 2),
            torch.nn.Conv2d(1, dim, 3, 2),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            # Conv2dLayerNorm(dim, dim, 3, 2),
            torch.nn.Conv2d(dim, dim, 3, 2),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(dim * (83 // 4), dim),
            # NOTE maybe required? but not converged?
            PositionalEncoding(dim, dropout)
        )

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # logging.warning("x {} mask {}".format(x.shape, x_mask.shape))
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Encoder(torch.nn.Module):
    def __init__(self, idim, args):
        super(Encoder, self).__init__()
        if args.input_layer == "linear":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Linear(idim, args.adim),
                torch.nn.Dropout(args.dropout_rate),
                torch.nn.ReLU(),
                # NOTE maybe required? but not converged
                PositionalEncoding(args.adim, args.dropout_rate)
            )
        elif args.input_layer == "conv2d":
            self.input_layer = Conv2dSubsampling(args.adim, args.dropout_rate)
        elif args.input_layer == "embed":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Embedding(idim, args.adim),
                PositionalEncoding(args.adim, args.dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + args.input_layer)

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
        if isinstance(self.input_layer, Conv2dSubsampling):
            x, mask = self.input_layer(x, mask)
        else:
            x = self.input_layer(x)
        x, mask = self.encoders(x, mask)
        return self.norm(x), mask


class Decoder(torch.nn.Module, ScoringBase):
    def __init__(self, odim, args):
        super(Decoder, self).__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(odim, args.adim),
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

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory_mask: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        x = self.output_layer(self.output_norm(x))
        return x, tgt_mask

    def init_state(self, h, enc_state, args):
        return enc_state

    def select_state(self, state, index):
        return state

    def score(self, token, enc_output, state):
        y, _ = self.forward(token, None, enc_output, state["mask"])
        return torch.log_softmax(y[:, -1, :], dim=-1), state


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduce=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum()  / total


class E2E(torch.nn.Module):
    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.encoder = Encoder(idim, args)
        self.decoder = Decoder(odim, args)
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = -1
        self.subsample = [0]
        # self.lsm_weight = a
        if args.lsm_weight > 0:
            self.criterion = LabelSmoothing(self.odim, self.ignore_id, args.lsm_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_id,
                                                 size_average=True)
        # self.char_list = args.char_list
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.recog_args = None  # unused

    def reset_parameters(self, args):
        if args.ninit == "none":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
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
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # embedding init
        self.decoder.embed[0].weight.data.normal_(0, 1)

    def add_sos_eos(self, ys_pad):
        from espnet.nets.e2e_asr_th import pad_list
        eos = ys_pad.new([self.eos])
        sos = ys_pad.new([self.sos])
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        return pad_list(ys_in, self.eos), pad_list(ys_out, self.ignore_id)

    def target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != self.ignore_id
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens)).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # forward decoder
        ys_in_pad, ys_out_pad = self.add_sos_eos(ys_pad)
        ys_mask = self.target_mask(ys_in_pad)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        self.pred_pad = pred_pad

        # compute loss
        loss_att = self.criterion(
            pred_pad.view(-1, self.odim),
            ys_out_pad.view(-1))
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                          ignore_label=self.ignore_id)

        # TODO(karita) show predected text
        # TODO(karita) calculate these stats
        device = xs_pad.device
        # acc = torch.as_tensor(acc).to(device)
        # loss_ctc = torch.as_tensor(0.0).to(device)
        # cer = torch.as_tensor(0.0).to(device)
        # wer = torch.as_tensor(0.0).to(device)
        loss_ctc = None
        cer, wer = 0.0, 0.0
        return loss_ctc, loss_att, acc, cer, wer

    def recognize(self, feat, recog_args, char_list=None, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        feat = torch.as_tensor(feat).unsqueeze(0)
        feat_len = [feat.size(1)]
        mask = (~make_pad_mask(feat_len)).to(feat.device).unsqueeze(-2)
        enc_output, mask = self.encoder(feat, mask)

        # TODO(karita) support CTC, LM, lpz
        if recog_args.beam_size == 1:
            logging.info("use greedy search implementation")
            ys = torch.full((1, 1), self.sos).long()
            score = torch.zeros(1)
            maxlen = feat.size(1) + 1
            for step in range(maxlen):
                ys_mask = subsequent_mask(step + 1).unsqueeze(0)
                out, _ = self.decoder(ys, ys_mask, enc_output, mask)
                prob = torch.log_softmax(out[:, -1], dim=-1)  # (batch, token)
                max_prob, next_id = prob.max(dim=1)  # (batch, token)
                score += max_prob
                if step == maxlen - 1:
                    next_id[0] = self.eos

                ys = torch.cat((ys, next_id.unsqueeze(1)), dim=1)
                if next_id[0].item() == self.eos:
                    break
            y = [{"score": score, "yseq": ys[0].tolist()}]
        else:
            search = BeamSearch([self.decoder], {self.decoder: 1.0}, self.sos, self.eos)
            y = search.recognize(enc_output, {"mask": mask}, recog_args, feat.device, char_list)
        self.training = prev
        return y


    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        '''E2E attention calculation
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            results = self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn
        return ret


def _plot_and_save_attention(att_w, filename):
    # dynamically import matplotlib due to not found error
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        # plt.subplot(1, len(att_w), h)
        ax.imshow(aw, aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(filename)


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png"):
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.%s.%s" % (
                outdir, data[idx][0], name, suffix)
            dec_len = int(data[idx][1]['output'][0]['shape'][0])
            enc_len = int(data[idx][1]['input'][0]['shape'][0])
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
            elif "decoder" in name:
                if "self" in name:
                    att_w = att_w[:, :dec_len, :dec_len]
                else:
                    att_w = att_w[:, :dec_len, :enc_len]
            else:
                logging.warning("unknown name for shaping attention")
            _plot_and_save_attention(att_w, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def __call__(self, trainer):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        attn_dict = self.att_vis_fn(*batch)
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        plot_multi_head_attention(self.data, attn_dict, self.outdir, suffix)
