from argparse import Namespace
import logging

import torch
import numpy

from espnet.nets.e2e_asr_common import end_detect


# interface for Decoder, CTC, RNNLM, etc
class ScoringBase(object):
    def score(self, token, h, state):
        '''scoring method
        :param torch.tensor token: token to score (B)
        :param torch.tensor h: encoded input features (B, T, D)
        :param dict enc_state: information of encoder to init decoders
        :param dict state: state of this scorer
        :return: log likelihood of token
        :rtype: torch.tensor
        '''
        pass

    def init_state(self, h, enc_state, args):
        '''initialize decoder state given by encoder and beam search setting
        :param torch.tensor h: encoded input features (B, T, D)
        :param dict enc_state: information of encoder to init decoders
        :param namespace args: beam search settting
        :return: initial decoder state
        :rtype: torch.tensor
        '''
        pass

    def select_state(self, state, index):
        '''select next decoder state given by previous state and index
        :param dict state: state to be selected
        :param int index: state index
        :return: next decoder state
        :rtype: dict
        '''
        pass


def local_prune(x, k):
    shape = x.shape
    n_token = shape[-1]
    v, ids = x.topk(k, dim=-1)
    mv, mk = v.min(dim=-1)
    mv = mv.unsqueeze(-1)
    masked = x.masked_fill(x < mv, 0)
    return v, ids, masked


def prune(n_beam, global_best_score, local_score, ended_mask):
    """prune to fit the beam
    :param int n_beam: the number of hypotheses in beam
    :return torch.Tensor global_best_score: global score of hypotheses (batch, beam)
    :return torch.Tensor local_score: local tokenwise score of hypotheses (batch, beam, token)
    :return torch.Tensor ended_mask: binary tensor to mask ended token {1:ended, 0:not ended} (batch, beam)

    :return torch.Tensor global_best_score: updated score of hypotheses (batch, beam)
    :return torch.Tensor global_best_id: selected hypotheses ids
    :return torch.Tensor global_token_id: selected token to be appended
    """
    n_batch, n_beam, n_token = local_score.shape

    # not to increase global score, mask ended hypotheses with zero
    local_score = local_score.masked_fill(ended_mask.unsqueeze(2), 0)

    # local prune (n_batch, n_beam, n_token) -> (n_batch, n_beam, n_beam)
    local_best_score, local_best_id, local_masked_score = local_prune(local_score, n_beam)

    # global prune (n_batch, n_beam, n_token) -> (n_batch, n_beam)
    global_score = global_best_score.reshape(n_batch, n_beam, 1) + local_masked_score
    global_score = global_score.reshape(n_batch, n_beam * n_token)
    global_best_score, global_best_id = global_score.topk(n_beam, -1)  # (n_batch, n_beam)
    global_best_hyp_id = global_best_id // n_token
    global_best_token_id = global_best_id % n_token
    return global_best_score, global_best_hyp_id, global_best_token_id


class BeamSearch(object):
    """chainer/pytorch model independent implementation of beam search"""
    def __init__(self, scorers, weight_dict, sos, eos):
        assert len(scorers) == len(weight_dict)
        for s in scorers:
            assert isinstance(s, ScoringBase)
        self.scorers = scorers
        self.weight_dict = weight_dict
        self.sos = sos
        self.eos = eos
        self.min_score = float(numpy.finfo(numpy.float32).min)

    # pytorch implementation
    def init_state(self, n_batch, n_beam, device):
        global_best_token = torch.full((n_batch, n_beam, 1), self.sos, dtype=torch.int64, device=device)
        global_best_score = torch.zeros((n_batch, n_beam), dtype=torch.float32, device=device)
        ended_mask = torch.full((n_batch, n_beam), False, dtype=torch.uint8, device=device)
        return global_best_token, global_best_score, ended_mask

    # pytorch implementation

    # pytorch implementation
    def append_tokens(self, global_best_token_id, global_best_tokens, eos_mask):
        cpu_best_token_id = global_best_token_id.cpu().numpy()
        cpu_eos_mask = eos_mask.cpu().numpy()
        for batch, beam_tokens in enumerate(global_best_tokens):
            for beam, tokens in enumerate(beam_tokens):
                if not cpu_eos_mask[batch, beam].item():
                    tokens.append(int(cpu_best_token_id[batch, beam]))

    def recognize(self, enc_output, enc_states, args, device, char_list=None):
        '''E2E beam search

        :param torch.tensor h: encoded input acouctic feature (B, T, D)
        :param dict h: information of h or encoder for decoders
        :param Namespace args: argment namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        '''
        logging.info("encoder output: {}".format(enc_output.shape))
        h = enc_output

        # prepare states
        states = dict()
        for s in self.scorers:
            states[s] = s.init_state(h, enc_states, args)

        n_batch, maxlen_in, n_feat = h.shape
        n_beam = args.beam_size
        global_best_token_seq = [[[] for _ in range(n_beam)] for _ in range(n_batch)]
        global_prev_token, global_best_score, eos_mask = self.init_state(n_batch, n_beam, device)
        h = h.unsqueeze(1).expand(n_batch, n_beam, maxlen_in, n_feat)  # (n_batch, n_hyp, maxlen_in, n_feat)

        if args.maxlenratio == 0:
            maxlen_out = maxlen_in
        else:
            maxlen_out = max(1, int(recog_args.maxlenratio * maxlen_in))
        minlen = int(args.minlenratio * maxlen_in)

        # prefix search
        ended_hyps = []
        for step in range(1, maxlen_out):
            print(step)
            # forward scorers
            local_score = 0
            new_scores = dict()
            new_states = dict()
            for s in self.scorers:
                new_scores[s], new_states[s] = s.score(
                    global_prev_token.view(-1, step),
                    h.view(-1, maxlen_in, n_feat),
                    states[s])
                # (n_batch, n_beam, n_token)
                local_score += self.weight_dict[s] * new_scores[s]

            # prune hypotheses
            n_token = local_score.size(-1)
            local_score = local_score.view(n_batch, -1, n_token)
            global_best_score, global_best_hyp_id, global_best_token \
                = prune(n_beam, global_best_score, local_score, eos_mask)

            # update results
            # self.append_tokens(global_best_token, global_best_token_seq, eos_mask)
            # print(global_best_hyp_id)
            # TODO(karita) batch support
            for i in range(n_batch):
                h[i] = h[i].index_select(0, global_best_hyp_id[i])
            for s in self.scorers:
                states[s] = s.select_state(new_states[s], global_best_hyp_id)

            if step == maxlen_out - 1:
                logging.info('adding <eos> in the last postion in the loop')
                global_best_token[:] = self.eos

            global_prev_token = torch.cat((global_prev_token, global_best_token.unsqueeze(-1)), dim=-1)
            eos_mask |= global_best_token == self.eos
            for ba in range(n_batch):
                for be in range(n_beam):
                    token = global_best_token[ba, be].item()
                    seq = global_best_token_seq[ba][be]
                    seq.append(token)
                    if token == self.eos and step >= minlen:
                        score = global_best_score[ba, be] + step * args.penalty
                        ended_hyps.append({"yseq": seq, "score": score})

            # TODO(karita) use end detection
            if torch.all(eos_mask):
                break  # no hypotheses remained

        # make N-best list
        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), args.nbest)]
        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            args = Namespace(**vars(args))
            args.minlenratio = max(0.0, args.minlenratio - 0.1)
            return self.recognize(enc_output, enc_states, args, device, char_list)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps
