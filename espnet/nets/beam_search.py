import numpy

# interface for Decoder, CTC, RNNLM, etc
class ScoringBase(object):
    def score(self, token, enc_output, state):
        pass

    def init_state(self, src, src_lengths, enc_output, args):
        pass

    def select_state(self, state, index):
        pass


class BeamSearch(object):
    """chainer/pytorch model independent implementation of beam search"""
    def __init__(self, encoder, scorers, weight_dict, sos, eos):
        assert len(scorers) == len(weight_dict)
        for s in scorers:
            assert isinstance(s, ScoringBase)
        self.encoder = encoder
        self.scorers = scorers
        self.weight_dict = weight_dict
        self.sos = sos
        self.eos = eos
        self.min_score = numpy.finfo(numpy.float32).min

    # pytorch implementation
    def init_state(self, n_batch, n_beam, device):
        global_best_token = torch.full((n_batch, 1), self.sos, dtype=torch.int64, device=device)
        global_best_score = torch.zeros((n_batch, n_beam), dtype=torch.float32, device=device)
        eos_mask = torch.full((n_batch, n_beam), dtype=torch.uint8, device=device)
        return global_best_token, global_best_score, eos_mask

    # pytorch implementation
    def prune(self, global_best_score, local_best_score, eos_mask):
        n_batch, n_beam, n_token = local_best_score.shape
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
    def update_tokens(self, global_best_token_id, global_best_tokens, eos_mask):
        cpu_best_token_id = global_best_token_id.cpu().numpy()
        cpu_eos_mask = eos_mask.cpu().numpy()
        for batch, beam_tokens in enumerate(global_best_tokens):
            for beam, tokens in enumerate(beam_tokens):
                if cpu_eos_mask[batch, beam] != 0:
                    tokens.append(int(cpu_best_token_id[batch, beam]))

    def recognize(self, src, src_length, args, device):
        # forward encoder
        enc_output = self.encoder(src, src_lengths)

        # prepare states
        states = dict()
        for s in self.scorers:
            states[s] = s.init_state(src, src_lengths, enc_output, args)

        n_batch = len(src_length)
        n_beam = args.beam_size
        global_best_token_seq = [[[] for _ in range(n_beam)] for _ in range(n_batch)]
        global_best_token, global_best_score, eos_mask = self.init_state(n_batch, n_beam, device)

        # prefix search
        for step in range(args.maxlen_decode):
            # forward scorers
            local_score = 0
            new_scores = dict()
            new_states = dict()
            for s in self.scorers:
                new_scores[s], new_states[s] = s.score(
                    global_best_token,
                    enc_output,
                    states[s])
                # (n_batch, n_beam, n_token)
                local_score += self.weight_dict[s] * new_scores[s]

            # prune hypotheses
            global_best_score, global_best_hyp_id, global_best_token \
                = self.prune(global_best_score, local_best_score, eos_mask)

            # update results
            self.update_tokens(global_best_token, global_best_token_seq, eos_mask)
            for s in self.scorers:
                states[s] = s.select_state(new_states[s], global_best_hyp_id)

            # end detection
            eos_mask |= batch_end_detect(global_best_score, global_best_tokens)
            if torch.all(eos_mask):
                break  # no hypotheses remained
        return global_best_tokens, global_best_score
