# coding: utf-8
# Copyright @tongshiwei

from tqdm import tqdm

import mxnet as mx
from gluonnlp.data import PadSequence
from longling.framework.ML.MXnet.util import split_and_load

try:
    from .EmbedDKTModule import Parameters, EmbedDKTModule
except ModuleNotFoundError:
    from EmbedDKTModule import Parameters, EmbedDKTModule


def load_net(load_epoch=Parameters.end_epoch, params=Parameters()):
    mod = EmbedDKTModule(params)

    net = mod.sym_gen(params.ku_num, params.hidden_num, params.latent_dim)
    net = mod.load(net, load_epoch, mod.params.ctx)
    return net


# todo 重命名eval_DKT函数到需要的模块名
def eval_DKT():
    pass


def batch_divide(datas, batch_size):
    batch = []
    cur_batch = []
    for data in datas:
        if len(cur_batch) >= batch_size:
            batch.append(cur_batch)
            cur_batch = []
        cur_batch.append(data)
    if cur_batch:
        batch.append(cur_batch)
    return batch


# todo 重命名use_DKT函数到需要的模块名
class DKT(object):
    """
    DKT模型

    References
    ----------
    .. code-block:: none

    @inproceedings{piech2015deep,
      title={Deep knowledge tracing},
      author={Piech, Chris and Bassen, Jonathan and Huang, Jonathan and Ganguli, Surya and Sahami, Mehran and Guibas, Leonidas J and Sohl-Dickstein, Jascha},
      booktitle={Advances in Neural Information Processing Systems},
      pages={505--513},
      year={2015}
    }

    Examples
    --------
    >>> dkt = DKT(Parameters())
    >>> a = [(123, 1), (123, 0), (155, 3)]
    >>> len(dkt(a))
    3
    """

    def __init__(self, params=None):
        self.net = load_net(params.end_epoch, params)
        self.params = params

    def __call__(self, responses, begin_state=None):
        """

        Parameters
        ----------
        responses: list[(int, int)]
            学生做题记录
        begin_state: list[float]

        Returns
        -------
        states: list[list[float]]
            学生在每一时刻的知识掌握情况

        Notes
        -----
        len(responses) == len(states), but states[0] 对应的是做完第一道题后的知识掌握情况

        """
        responses = [self.one_hot(r) for r in responses]
        begin_state = [mx.nd.array([_], ctx=self.params.ctx) for _ in begin_state] if begin_state else None
        mastery, state = self.net(mx.nd.array([responses], ctx=self.params.ctx), None, True, begin_state)
        mastery = mastery.asnumpy().tolist()
        state = [_.asnumpy().tolist()[0] for _ in state]
        return mastery[0], state

    def one_hot(self, r):
        correct = 0 if r[1] <= 0 else 1
        return r[0] * 2 + correct


if __name__ == '__main__':
    # a = mx.nd.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    # print(mx.nd.pick(a, mx.nd.array([0, 1, 2])))
    # dkt = DKT(Parameters(batch_size=128))
    from mxnet import gpu,cpu
    # sim_dkt = DKT(Parameters(model_name="SimDKT", ctx=gpu(1)))
    sim_dkt = DKT(Parameters(model_name="EmbedDkt", ctx=cpu(1)))
    a = [(123, 1), (123, 0), (155, 3)]
    # print(dkt(a, [0] * 835))
    print(sim_dkt(a, [0] * 10))
