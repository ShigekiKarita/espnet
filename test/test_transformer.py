import torch

import espnet.nets.e2e_asr_transformer_th as T

from argparse import Namespace
import logging
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')


def test_sequential():
    class Masked(torch.nn.Module):
        def forward(self, x, m):
            return x, m

    f = T.MultiSequential(Masked(), Masked())
    x = torch.randn(2, 3)
    m = torch.randn(2, 3) > 0
    assert len(f(x, m)) == 2
    if torch.cuda.is_available():
        f = torch.nn.DataParallel(f)
        f.cuda()
        assert len(f(x.cuda(), m.cuda())) == 2


def test_mask():
    m = T.subsequent_mask(3)
    assert m.tolist() == [[[1, 1, 1], [0, 1, 1], [0, 0, 1]]]


def prepare():
    args = Namespace(
        adim=64,
        aheads=8,
        dropout_rate=0.,
        elayers=2,
        eunits=32,
        dlayers=2,
        dunits=32,
        ninit="none"
    )
    idim = 3
    odim = 4
    model = T.E2E(idim, odim, args)

    x = torch.randn(5, 70, idim)
    ilens = [70, 50, 30, 30, 20]
    y = (torch.rand(5, 10) * odim % odim).long()
    olens = [3, 9, 10, 2, 3]
    for i in range(x.size(0)):
        x[i, ilens[i]:] = 0
        y[i, olens[i]:] = model.ignore_id

    data = []
    for i in range(x.size(0)):
        data.append(("utt%d" % i, {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}]
        }))
    return model, x, ilens, y, data


def test_transformer():
    model, x, ilens, y, data = prepare()

    # test acc is almost 100%
    optim = torch.optim.Adam(model.parameters(), 0.01)
    for i in range(10):
        loss_ctc, loss_att, acc, cer, wer = model(x, ilens, y)
        optim.zero_grad()
        loss_att.backward()
        optim.step()
        print(loss_att, acc)
        # attn_dict = model.calculate_all_attentions(x, ilens, y)
        # T.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test", "iter%d.png" % i)
    assert acc > 0.9

    # # test attention plot
    # attn_dict = model.calculate_all_attentions(x, ilens, y)
    # T.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test")

    # test beam search
    recog_args = Namespace(
        beam_size=3,
        penalty=0.1,
        ctc_weight=0.0,
        maxlenratio=0,
        minlenratio=0,
    )
    with torch.no_grad():
        model.recognize(x[0, :ilens[0]].numpy(), recog_args)


def test_transformer_parallel():
    if not torch.cuda.is_available():
        return

    class LossAcc(torch.nn.Module):
        def __init__(self, model):
            super(LossAcc, self).__init__()
            self.model = model

        def forward(self, *args):
            loss_ctc, loss_att, acc, cer, wer = self.model(*args)
            return loss_att, torch.as_tensor(acc).to(loss_att.device)

    model, x, ilens, y, data = prepare()
    model = LossAcc(model)
    model = torch.nn.DataParallel(model).cuda()
    logging.debug(ilens)
    # test acc is almost 100%
    optim = torch.optim.Adam(model.parameters(), 0.01)
    for i in range(10):
        loss_att, acc = model(x, torch.as_tensor(ilens), y)
        optim.zero_grad()
        acc = acc.mean().item()
        loss_att.mean().backward()
        optim.step()
        print(loss_att, acc)
        # attn_dict = model.calculate_all_attentions(x, ilens, y)
        # T.plot_multi_head_attention(data, attn_dict, "/tmp/espnet-test", "iter%d.png" % i)
    assert acc > 0.9



if __name__ == "__main__":
    # test_transformer()
    test_transformer_parallel()
