# -*- coding: utf-8 -*-
import os
import re
import json
from typing import Dict, List, Optional,Tuple,Union

class ScaleAgent:
    """
    量表代理类，负责所有量表相关的功能：
    1. 加载和管理量表数据
    2. 解析用户回答
    3. 评估量表结果
    4. 生成专业建议
    """


