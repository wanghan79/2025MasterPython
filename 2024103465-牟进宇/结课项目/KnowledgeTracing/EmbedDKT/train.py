# coding: utf-8
# Copyright @tongshiwei


from longling.lib.clock import Clock
from longling.lib.utilog import config_logging
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, Evaluator, MovingLosses, \
    ClassEvaluator
from longling.framework.ML.MXnet.viz import plot_network, VizError


"""
更新日志
20231028
为运行assistments12,更改了输入数据路径
"""


try:
    from .EmbedDKTModule import *
except (SystemError, ModuleNotFoundError):
    from EmbedDKTModule import *


def train_EDKT(train_data_path, test_data_path, **kwargs):
    """
    训练 EmbedDKT 模型的函数。

    :param train_data_path: 训练数据文件的路径，相对于参数中定义的数据目录
    :param test_data_path: 测试数据文件的路径，相对于参数中定义的数据目录
    :param kwargs: 额外的关键字参数，用于初始化参数类
    :return: 无
    """
    # 1 配置参数初始化
    # 使用传入的关键字参数初始化 Parameters 类实例
    params = Parameters(
        **kwargs
    )

    # 创建 EmbedDKTModule 实例，并传入参数
    mod = EmbedDKTModule(params)
    # 记录模块信息
    mod.logger.info(str(mod))

    # 从参数中获取批量大小、起始轮数、结束轮数和计算设备
    batch_size = mod.params.batch_size
    begin_epoch = mod.params.begin_epoch
    end_epoch = mod.params.end_epoch
    ctx = mod.params.ctx

    # 2.1 重新生成
    # 记录开始生成网络符号的信息
    mod.logger.info("generating symbol")
    # 调用模块的 sym_gen 方法生成网络符号
    net = mod.sym_gen(params.ku_num, params.hidden_num, params.latent_dim)

    # 2.2 装载已有模型
    # 以下代码为注释示例，实际未执行
    # net = mod.load(epoch)
    # net = DKTModule.load_net(filename)

    # 3 可视化检查网络
    # 以下代码为注释示例，实际未执行
    # net_viz(net, mod.params)

    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数, 命名中不能出现 下划线
    # 定义反向传播的损失函数
    bp_loss_f = {
        "loss": DKTLoss(),
    }
    # 初始化损失函数字典
    loss_function = {

    }
    # 将反向传播损失函数添加到损失函数字典中
    loss_function.update(bp_loss_f)
    # 创建损失监控器，用于监控训练过程中的损失
    losses_monitor = MovingLosses(loss_function)

    # 创建计时器，用于记录训练时间
    timer = Clock()
    # 创建训练批次信息提示器，用于在训练过程中提示信息
    informer = TrainBatchInformer(loss_index=[name for name in loss_function], end_epoch=params.end_epoch - 1)
    # 配置验证日志记录器
    validation_logger = config_logging(
        filename=params.model_dir + "result.log",
        logger="%s-validation" % params.model_name,
        mode="w",
        log_format="%(message)s",
    )
    # 导入评估指标类
    from longling.framework.ML.MXnet.metric import PRF, Accuracy
    # 创建评估器，用于评估模型性能
    evaluator = ClassEvaluator(
        metrics=[PRF(argmax=False), Accuracy(argmax=False)],
        model_ctx=ctx,
        logger=validation_logger,
        log_f=mod.params.validation_result_file
    )

    # 记录开始加载数据的信息
    mod.logger.info("loading data")
    # 加载训练数据
    train_data = EmbedDKTModule.get_data_iter(params.data_dir + train_data_path, params)
    # 加载测试数据
    test_data = EmbedDKTModule.get_data_iter(params.data_dir + test_data_path, params)

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    # 记录开始训练的信息
    mod.logger.info("start training")
    try:
        # 尝试从已有模型文件中加载网络参数
        net = mod.load(net, begin_epoch, params.ctx)
        mod.logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
    except FileExistsError:
        # 若模型文件不存在，则初始化网络
        mod.logger.info("model doesn't exist, initializing")
        EmbedDKTModule.net_initialize(net, ctx)
    # 创建训练器，用于训练网络
    trainer = EmbedDKTModule.get_trainer(net, optimizer=params.optimizer, optimizer_params=params.optimizer_params)
    # todo whether to use static symbol to accelerate, do not invoke this method for dynamic structure like rnn
    # 以下代码为注释示例，实际未执行
    # net.hybridize()
    # 再次记录开始训练的信息
    mod.logger.info("start training")
    # 调用模块的 fit 方法开始训练模型
    mod.fit(
        net=net, begin_epoch=begin_epoch, end_epoch=end_epoch, batch_size=batch_size,
        train_data=train_data,
        trainer=trainer, bp_loss_f=bp_loss_f,
        loss_function=loss_function, losses_monitor=losses_monitor,
        test_data=test_data,
        ctx=ctx,
        informer=informer, epoch_timer=timer, evaluator=evaluator,
        prefix=mod.prefix,
        save_epoch=params.save_epoch,
    )
    # 以下代码需要在调用 hybridize 方法并至少 forward 一次后执行
    # net.export(mod.prefix)  # 需要在这之前调用 hybridize 方法,并至少forward一次

    # optional todo 评估
    # 以下代码为注释示例，实际未执行
    # DKTModule.eval()


if __name__ == '__main__':
    from mxnet import gpu,cpu

    # train_EDKT("DKT/train", "DKT/test", dataset="ifly")
    # train_EDKT("DKT/sim_train", "DKT/sim_test", model_name="SimEDKT", ctx=gpu(1), dataset="ifly")

    # train_EDKT("train", "test", ctx=gpu(0), dataset="junyi")
    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(1), dataset="junyi")

    # train_EDKT("train", "test", ctx=gpu(0), dataset="junyi_50")
    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(0), dataset="junyi_50")

    # train_EDKT("train", "test", ctx=gpu(2), dataset="junyi_80")
    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(3), dataset="junyi_80")

    # train_EDKT("train", "test", ctx=gpu(0), dataset="junyi_long")
    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(1), dataset="junyi_long")

    # train_EDKT("train", "test", ctx=gpu(0), dataset="junyi_large")
    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(1), dataset="junyi_large")

    # train_EDKT("train", "test", ctx=gpu(1), dataset="junyi_session")
    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(0), dataset="junyi_session")

    #20231028以前
    # train_EDKT("train", "test", ctx=cpu(1), dataset="simirt")
    #20231028为运行assistments12修改数据集和文件名
    #####train_EDKT("assistments2012_dkt_train_dataset", "assistments2012_dkt_test_dataset", ctx=cpu(), dataset="assistments12")

    # train_EDKT("sim_train", "sim_test", model_name="SimEDKT", ctx=gpu(3), dataset="simirt")

    # train_EDKT("train", "test", ctx=gpu(2), dataset="irt")

    # train_EDKT("train", "test", ctx=cpu(1), dataset="kss")

    # train_EDKT("train", "test", ctx=gpu(0), dataset="assistments")
    #
    # train_EDKT("train", "test", ctx=gpu(1), dataset="a0910c")

    # train_EDKT("train", "test", ctx=gpu(2), dataset="assistments100")
    #
    # train_EDKT("train", "test", ctx=gpu(3), dataset="a0910c100")
    #
    # train_EDKT("train", "test", ctx=gpu(3), dataset="a0910c100loop")
    train_EDKT("dkt_train","dkt_test", ctx=gpu(0), dataset="junyi")