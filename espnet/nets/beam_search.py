import torch
import numpy

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
        self.min_score = numpy.finfo(numpy.float32).min

    # pytorch implementation
    def init_state(self, n_batch, n_beam, device):
        global_best_token = torch.full((n_batch, 1), self.sos, dtype=torch.int64, device=device)
        global_best_score = torch.zeros((n_batch, n_beam), dtype=torch.float32, device=device)
        eos_mask = torch.full((n_batch, n_beam), False, dtype=torch.uint8, device=device)
        return global_best_token, global_best_score, eos_mask

    # pytorch implementation
    def prune(self, global_best_score, local_score, eos_mask):
        n_batch, n_beam, n_token = local_score.shape
        # not to increase global score, mask ended hypotheses with zero
        local_score.mul_(eos_mask.unsqueeze(-1).float())
        n_token = token_score.shape[-1]
        # local prune (n_batch, n_beam, n_token) -> (n_batch, n_beam, n_beam)
        local_worst_score, local_worst_id = torch.topk(-token_score, n_token - n_beam, -1)
        # not to select in global prune, mask tokens with min value except for top n_beam
        local_score.index_fill_(-1, local_worst_id, self.min_score)
        # global prune (n_batch, n_beam, n_beam) -> (n_batch, n_beam)
        global_score = global_best_score.reshape(n_batch, n_beam, 1) + local_best_score
        global_score = global_score.reshape(n_batch, n_beam * n_beam)
        global_best_score, global_best_id = topk(global_score, n_beam, -1)  # (n_batch, n_beam)
        global_best_hyp_id = global_best_id // n_beam
        global_best_token_id = global_best_id % n_beam
        return global_best_score, global_best_hyp_id, global_token_id

    # pytorch implementation
    def append_tokens(self, global_best_token_id, global_best_tokens, eos_mask):
        cpu_best_token_id = global_best_token_id.cpu().numpy()
        cpu_eos_mask = eos_mask.cpu().numpy()
        for batch, beam_tokens in enumerate(global_best_tokens):
            for beam, tokens in enumerate(beam_tokens):
                if cpu_eos_mask[batch, beam]:
                    tokens.append(int(cpu_best_token_id[batch, beam]))

    def recognize(self, h, enc_states, args, device, char_list=None):
        '''E2E beam search

        :param torch.tensor h: encoded input acouctic feature (B, T, D)
        :param dict h: information of h or encoder for decoders
        :param Namespace args: argment namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        '''

        # prepare states
        states = dict()
        for s in self.scorers:
            states[s] = s.init_state(h, enc_states, args)

        n_batch, maxlen_in, n_feat = h.shape
        n_beam = args.beam_size
        global_best_token_seq = [[[] for _ in range(n_beam)] for _ in range(n_batch)]
        global_best_token, global_best_score, eos_mask = self.init_state(n_batch, n_beam, device)
        h = h.unsqueeze(1)  # (n_batch, n_hyp, maxlen_in, n_feat)

        if args.maxlenratio == 0:
            maxlen_out = maxlen_in
        else:
            maxlen_out = max(1, int(recog_args.maxlenratio * maxlen_in))

        # prefix search
        for step in range(maxlen_out):
            # forward scorers
            local_score = 0
            new_scores = dict()
            new_states = dict()
            for s in self.scorers:
                new_scores[s], new_states[s] = s.score(
                    global_best_token.view(-1),
                    h.view(-1, maxlen_in, n_feat),
                    states[s])
                # (n_batch, n_beam, n_token)
                local_score += self.weight_dict[s] * new_scores[s]

            # prune hypotheses
            n_token = local_score.size(-1)
            local_score = local_score.view(n_batch, -1, n_token)
            global_best_score, global_best_hyp_id, global_best_token \
                = self.prune(global_best_score, local_score, eos_mask)

            # update results
            self.append_tokens(global_best_token, global_best_token_seq, eos_mask)
            h = h.index_select(1, global_best_hyp_id)
            for s in self.scorers:
                states[s] = s.select_state(new_states[s], global_best_hyp_id)

            # end detection
            eos_mask |= batch_end_detect(global_best_score, global_best_tokens)
            if torch.all(eos_mask):
                break  # no hypotheses remained
        return global_best_tokens, global_best_score
