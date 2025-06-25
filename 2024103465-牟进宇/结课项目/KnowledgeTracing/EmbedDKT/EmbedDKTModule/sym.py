# coding: utf-8
# Copyright @tongshiwei

import logging
import os

import mxnet as mx
from mxnet import gluon

from longling.framework.ML.MXnet.viz import plot_network, VizError
from longling.framework.ML.MXnet.mx_gluon.gluon_util.sequence import format_sequence


class DKTNet(gluon.HybridBlock):
    def __init__(self, ku_num, hidden_num, latent_dim, **kwargs):
        super(DKTNet, self).__init__(**kwargs)

        self.length = None

        with self.name_scope():
            self.embedding = gluon.nn.Embedding(2 * ku_num, latent_dim)
            self.embedding_dropout = gluon.nn.Dropout(0.2)
            self.lstm = gluon.rnn.HybridSequentialRNNCell()
            self.lstm.add(
                gluon.rnn.LSTMCell(hidden_num),
            )
            self.dropout = gluon.nn.Dropout(0.5)
            self.nn = gluon.nn.Dense(ku_num)
            # self.lstm.add(
            #     gluon.rnn.LSTMCell(ku_num * 8),
            # )
            # self.lstm.add(gluon.rnn.DropoutCell(0.5))
            # self.lstm.add(
            #     gluon.rnn.LSTMCell(ku_num)
            # )

    def hybrid_forward(self, F, responses, mask=None, merge_outputs=True, begin_state=None, *args, **kwargs):
        length = self.length if self.length else len(responses[0])

        outputs, states = self.lstm.unroll(length, self.embedding_dropout(self.embedding(responses)),
                                           begin_state=begin_state, merge_outputs=False, valid_length=mask)

        outputs = [self.nn(self.dropout(output)) for output in outputs]
        output, _, _, _ = format_sequence(length, outputs, 'NTC', merge=merge_outputs)
        output = F.sigmoid(output)
        return output, states


class DKTLoss(gluon.HybridBlock):
    """
    Notes
    -----
    The loss has been average, so when call the step method of trainer, batch_size should be 1
    """
    def __init__(self, **kwargs):
        super(DKTLoss, self).__init__(**kwargs)

        with self.name_scope():
            self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
            # self.loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

    def hybrid_forward(self, F, pred_rs, pick_index, label, label_mask, *args, **kwargs):
        pred_rs = F.slice(pred_rs, (None, None), (None, -1))
        pred_rs = F.pick(pred_rs, pick_index)
        weight_mask = F.squeeze(
            F.SequenceMask(F.expand_dims(F.ones_like(pred_rs), -1), sequence_length=label_mask,
                           use_sequence_length=True, axis=1)
        )
        loss = self.loss(pred_rs, label, weight_mask)
        # loss = F.sum(loss, axis=-1)
        loss = F.mean(loss)
        return loss


def net_viz(net, params, **kwargs):
    batch_size = params.batch_size
    model_dir = params.model_dir
    logger = kwargs.get('logger', params.logger if hasattr(params, 'logger') else logging)

    try:
        view_tag = params.view_tag
    except AttributeError:
        view_tag = True

    try:
        viz_dir = os.path.abspath(model_dir + "plot/network")
        logger.info("visualization: file in %s" % viz_dir)
        from copy import deepcopy

        viz_net = deepcopy(net)
        viz_net.length = 2
        viz_shape = {'data': (batch_size,) + (2, )}
        x = mx.sym.var("data")
        sym = viz_net(x)[1][-1]
        plot_network(
            nn_symbol=sym,
            save_path=viz_dir,
            shape=viz_shape,
            node_attrs={"fixedsize": "false"},
            view=view_tag
        )
    except VizError as e:
        logger.error("error happen in visualization, aborted")
        logger.error(e)


def get_data_iter(params):
    """
    处理方式1（使用）

    对 hint（-1） 的处理, 看作做题（输入）中的正确，label上的不正确，
    即这道题不会（看了hint），但是达到了学会的效果（看了答案）

    处理方式2

    输入增加一维
    计算loss的时候使用mask屏蔽掉

    处理方式3

    新加一个标签，有冲突，不太可行

    """

    import random
    from gluonnlp.data import FixedBucketSampler, PadSequence

    num_buckets = params.num_buckets
    batch_size = params.batch_size
    ku_num = params.ku_num

    random.seed(10)

    responses = [[(random.randint(0, ku_num - 1), random.randint(-1, 1)) for _ in range(random.randint(2, 20))] for _
                 in range(1000)]

    responses.sort(key=lambda x: len(x))

    batch_idxes = FixedBucketSampler([len(rs) for rs in responses], batch_size, num_buckets=num_buckets)
    batch = []

    def one_hot(r):
        correct = 0 if r[1] < 0 else 1
        return r[0] * 2 + correct

    for batch_idx in batch_idxes:
        batch_rs = []
        batch_pick_index = []
        batch_labels = []
        for idx in batch_idx:
            batch_rs.append([one_hot(r) for r in responses[idx]])
            if len(responses[idx]) <= 1:
                pick_index, labels = [], []
            else:
                pick_index, labels = zip(*[(r[0], 0 if r[1] <= 0 else 1) for r in responses[idx][1:]])
            batch_pick_index.append(list(pick_index))
            batch_labels.append(list(labels))

        max_len = max([len(rs) for rs in batch_rs])
        padder = PadSequence(max_len, pad_val=0)
        batch_rs, data_mask = zip(*[(padder(rs), len(rs)) for rs in batch_rs])

        max_len = max([len(rs) for rs in batch_labels])
        padder = PadSequence(max_len, pad_val=0)
        batch_labels, label_mask = zip(*[(padder(labels), len(labels)) for labels in batch_labels])
        batch_pick_index = [padder(pick_index) for pick_index in batch_pick_index]
        batch.append(
            [mx.nd.array(batch_rs), mx.nd.array(data_mask), mx.nd.array(batch_labels), mx.nd.array(batch_pick_index),
             mx.nd.array(label_mask)])

    return batch


if __name__ == '__main__':
    # # set parameters
    try:
        # for python module
        from .parameters import Parameters
    except (ImportError, SystemError):
        # for python script
        from parameters import Parameters

    params = Parameters()

    # # set batch size
    # batch_size = 128
    # params = Parameters(batch_size=batch_size)
    #
    # generate sym
    net = DKTNet(params.ku_num, params.hidden_num, params.latent_dim)

    # visualiztion check
    params.view_tag = True
    net_viz(net, params)
    #
    # # numerical check
    # datas = get_data_iter()
    from tqdm import tqdm

    net.initialize()

    # bp_loss_f = lambda x, y: (x, y)
    # for data, label in tqdm(get_data_iter(params)):
    #     loss = bp_loss_f(net(data), label)
    import numpy as np
    from mxnet import autograd

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    bp_loss_f = DKTLoss()
    for epoch in range(0, 100):
        epoch_loss = 0
        for data, data_mask, label, pick_index, label_mask in get_data_iter(params):
            with autograd.record():
                pred_rs, _ = net(data, data_mask)
                loss = bp_loss_f(pred_rs, pick_index, label, label_mask)
                epoch_loss += loss.asscalar()
                # epoch_loss += np.mean(loss.asnumpy())
                loss.backward()
            trainer.step(1)
        print(epoch_loss)
