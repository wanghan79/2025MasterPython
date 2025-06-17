# coding: utf-8
# Copyright @tongshiwei

import argparse
import inspect
import os
import datetime

import yaml

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.stream import wf_open
from longling.framework.ML.MXnet.mx_gluon.glue.parser import MXCtx

from mxnet import cpu, gpu, lr_scheduler

"""
20231028
为测试assistments12数据集，添加适配assistments12的参数文件
20231028
更改批量大小为1024,epoch为50
20231028-2
更改批量大小为16
"""
class Parameters(object):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")) + os.sep
    model_name = os.path.basename(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    # Can also specify the model_name using explicit string
    # model_name = "DKT"

    # dataset = "junyi"
    dataset = "simirt"

    if dataset:
        root_data_dir = os.path.abspath(os.path.join(root, "data")) + os.sep
    else:
        root_data_dir = os.path.abspath(os.path.join(root, "data" + os.sep + "{}".format(dataset))) + os.sep

    data_dir = os.path.abspath(os.path.join(root_data_dir, "data")) + os.sep
    model_dir = os.path.abspath(os.path.join(root_data_dir, model_name)) + os.sep

    time_stamp = False
    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)

    optimizer = 'adam'
    optimizer_params = {
        'learning_rate': 0.001,
        # 'lr_scheduler': lr_scheduler.FactorScheduler(100, 0.9, 0.001),
        # 'wd': 0.5,
        # 'momentum': 0.01,
        # 'clip_gradient': 5,
    }

    begin_epoch = 0

    # end_epoch = 20
    #20231028改为50
    end_epoch = 20


    #20231028改为1024
    # batch_size = 16
    #batch_size = 1024
    #20231028-2对比实验，恢复为16
    batch_size = 16

    ctx = gpu(0)

    # 参数保存频率
    save_epoch = 1

    # 是否显示网络图
    view_tag = False

    # 更新保存参数，一般需要保持一致
    train_select = None
    save_select = train_select

    # 用户参数
    ku_num_dict = {
        "junyi": 835,
        "junyi_50": 835,
        "junyi_80": 835,
        "junyi_long": 835,
        "junyi_large": 835,
        "junyi_session": 835,
        "ifly": 945,
        "simirt": 10,
        "irt": 10,
        'kss': 10,

        #HYZ_20231028
        "assistments12":265,

        "assistments": 124,
        "assistments100": 124,
        "a0910c": 146,
        "a0910c100": 146,
        "a0910c100loop": 146,
    }
    hidden_num_dict = {
        "junyi": 900,
        "junyi_50": 900,
        "junyi_80": 900,
        "junyi_long": 900,
        "junyi_large": 900,
        "junyi_session": 900,
        "ifly": 1000,
        "simirt": 20,
        "irt": 20,
        'kss': 20,
        "assistments": 200,
        "assistments100": 200,

        #HYZ_20231028
        "assistments12": 375,

        "a0910c": 200,
        "a0910c100": 200,
        "a0910c100loop": 200,
    }
    latent_dim_dict = {
        "junyi": 600,
        "junyi_50": 600,
        "junyi_80": 600,
        "junyi_long": 600,
        "junyi_large": 600,
        "junyi_session": 600,
        "ifly": 650,
        "simirt": 15,
        "irt": 15,
        'kss': 15,
        "assistments": 75,
        "assistments100": 75,

        #HYZ_20231028
        "assistments12": 115,

        "a0910c": 85,
        "a0910c100": 85,
        "a0910c100loop": 85,
    }
    num_buckets = 100

    def __init__(self, params_yaml=None, **kwargs):
        """
        初始化 Parameters 类的实例。

        :param params_yaml: YAML 文件的路径，包含要加载的参数，默认为 None。
        :param kwargs: 额外的关键字参数，用于更新参数。
        """
        # 获取类变量，将其作为初始参数
        params = self.class_var
        # 默认数据集设置为 "junyi"
        self.dataset = "junyi"
        # 如果提供了 YAML 文件路径，则从文件中加载参数并更新到 params 中
        if params_yaml:
            params.update(self.load(params_yaml=params_yaml))
        # 使用传入的关键字参数更新 params
        params.update(**kwargs)
        # 将 params 中的参数逐个设置为实例属性
        for param, value in params.items():
            setattr(self, "%s" % param, value)
        # 根据传入的关键字参数重建相关的目录或文件路径
        if 'root_data_dir' not in kwargs:
            # 如果 dataset 为空，则根数据目录为项目根目录下的 data 目录
            if not self.dataset:
                self.root_data_dir = os.path.abspath(os.path.join(self.root, "data")) + os.sep
            # 否则，根数据目录为项目根目录下的 data 目录下以 dataset 命名的子目录
            else:
                self.root_data_dir = os.path.abspath(
                    os.path.join(self.root, "data") + os.sep + "{}".format(self.dataset)) + os.sep
        # 如果未传入 data_dir 参数，则重新计算数据目录
        if 'data_dir' not in kwargs:
            self.data_dir = os.path.abspath(os.path.join(self.root_data_dir, "data")) + os.sep
        # 如果未传入 model_dir 参数，则重新计算模型目录
        if 'model_dir' not in kwargs:
            # 如果实例有 time_stamp 属性且为 True，同时有 model_dir 属性，则在模型目录后添加时间戳
            if hasattr(self, 'time_stamp') and self.time_stamp and hasattr(self, 'model_dir'):
                time_stamp = "_%s" % datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                self.model_dir = os.path.abspath(
                    os.path.join(self.root_data_dir, self.model_name)) + time_stamp + os.sep
            # 否则，使用常规的模型目录
            else:
                self.model_dir = os.path.abspath(os.path.join(self.root_data_dir, self.model_name)) + os.sep
        # 验证结果文件的路径，位于模型目录下的 result.json
        self.validation_result_file = os.path.abspath(os.path.join(self.model_dir, "result.json"))
        # 根据当前数据集从 ku_num_dict 中获取知识单元数量
        self.ku_num = self.ku_num_dict[self.dataset]
        # 根据当前数据集从 hidden_num_dict 中获取隐藏层数量
        self.hidden_num = self.hidden_num_dict[self.dataset]
        # 根据当前数据集从 latent_dim_dict 中获取潜在维度数量
        self.latent_dim = self.latent_dim_dict[self.dataset]

    def items(self):
        return {k: v for k, v in vars(self).items() if k not in {'logger'}}

    @staticmethod
    def load(params_yaml, logger=None):
        f = open(params_yaml)
        params = yaml.load(f)
        if 'ctx' in params:
            params['ctx'] = MXCtx.load(params['ctx'])
        f.close()
        if logger:
            params['logger'] = logger
        return params

    def __str__(self):
        return str(self.parsable_var)

    def dump(self, param_yaml, override=False):
        if os.path.isfile(param_yaml) and not override:
            self.logger.warning("file %s existed, dump aborted" % param_yaml)
            return
        self.logger.info("writing parameters to %s" % param_yaml)
        with wf_open(param_yaml) as wf:
            dump_data = yaml.dump(self.parsable_var, default_flow_style=False)
            print(dump_data, file=wf)
            self.logger.info(dump_data)

    @property
    def class_var(self):
        variables = {k: v for k, v in vars(type(self)).items() if
                     not inspect.isroutine(v) and k not in self.excluded_names()}
        return variables

    @property
    def parsable_var(self):
        store_vars = {k: v for k, v in vars(self).items() if k not in {'logger'}}
        if 'ctx' in store_vars:
            store_vars['ctx'] = MXCtx.dump(store_vars['ctx'])
        return store_vars

    @staticmethod
    def excluded_names():
        """
        获取非参变量集
        Returns
        -------
        exclude names set: set
            所有非参变量
        """
        return {'__doc__', '__module__', '__dict__', '__weakref__',
                'class_var', 'parsable_var'}


class ParameterParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ParameterParser, self).__init__(*args, **kwargs)
        self.add_argument('--root_prefix', dest='root_prefix', default='', help='set root prefix')
        params = {k: v for k, v in vars(Parameters).items() if
                  not inspect.isroutine(v) and k not in Parameters.excluded_names()}
        for param, value in params.items():
            if param == 'logger':
                continue
            self.add_argument('--%s' % param, help='set %s, default is %s' % (param, value), default=value)
        self.add_argument('--kwargs', required=False, help=r"add extra argument here, use format: <key>=<value>")

    @staticmethod
    def parse(arguments):
        arguments = vars(arguments)
        args_dict = dict()
        for k, v in arguments.items():
            if k in {'root_prefix'}:
                continue
            args_dict[k] = v
        if arguments['root_prefix']:
            args_dict['root'] = os.path.abspath(os.path.join(arguments['root_prefix'], arguments['root']))

        return args_dict


if __name__ == '__main__':
    default_yaml_file = os.path.join(Parameters.model_dir, "parameters.yaml")

    # 命令行参数配置
    parser = ParameterParser()
    kwargs = parser.parse_args()
    kwargs = parser.parse(kwargs)

    data_dir = os.path.join(Parameters.root, "data")
    parameters = Parameters(
        **kwargs
    )
    parameters.dump(default_yaml_file, override=True)
    try:
        logger = parameters.logger
        parameters.load(default_yaml_file, logger=logger)
        parameters.logger.info('format check done')
    except Exception as e:
        print("parameters format error, may contain illegal data type")
        raise e
