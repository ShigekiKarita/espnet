import numpy
import torch
from espnet.nets.beam_search import local_prune, prune


def test_prune():
    n_batch = 2
    n_beam = 3
    n_token = 4
    g = torch.tensor([[1, 2, 3], [3, 2, 1]]).float()  # (batch, beam)
    l = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]],
        [[0.4, 0.1, 0.2, 0.3], [0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.4, 0.1]]
    ]).float()  # (batch, beam, token)

    # test local prune
    v, k, lm = local_prune(l, n_beam)
    l_pruned = torch.tensor([
        [[0.0, 0.2, 0.3, 0.4],  # + 1
         [0.2, 0.3, 0.4, 0.0],  # + 2
         [0.3, 0.4, 0.0, 0.2]], # + 3 but ended
        [[0.4, 0.0, 0.2, 0.3],  # + 3
         [0.4, 0.3, 0.2, 0.0],  # + 2 but ended
         [0.3, 0.2, 0.4, 0.0]]  # + 1 but ended
    ])
    numpy.testing.assert_equal(lm.tolist(), l_pruned.tolist())

    # test global prune
    e = torch.tensor([[0, 0, 1], [0, 1, 1]]).byte()  # (batch, beam)
    s, i, t = prune(n_beam, g, l, e)  # (batch, beam)
    print(i)
    numpy.testing.assert_equal(
        s.tolist(),
        torch.tensor(
            [[3.0, 3.0, 3.0],
             [3.4, 3.3, 3.2]]).tolist())
    print(t)  # (batch, beam)
    numpy.testing.assert_equal(i.tolist(), [[2,2,2], [0,0,0]])  # where 3 was selected from g
    numpy.testing.assert_equal(t.tolist(), [[1,0,3], [0,3,2]])  # where 0.4 was selected from l

