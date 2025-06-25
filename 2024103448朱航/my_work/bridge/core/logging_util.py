#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import codecs

class UnicodeStreamHandler(logging.StreamHandler):
    """支持Unicode的日志处理器"""
    
    def __init__(self, stream=None, encoding='utf-8'):
        logging.StreamHandler.__init__(self, stream)
        self.encoding = encoding
        
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                if isinstance(msg, unicode) and getattr(stream, 'encoding', None):
                    stream.write(msg.encode(self.encoding))
                else:
                    stream.write(msg)
            except UnicodeError:
                stream.write(msg.encode('ascii', 'replace'))
            stream.write(os.linesep)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class LoggingUtil(object):
    """日志工具类"""
    
    @staticmethod
    def configure_logger(name="PepperRobot", debug=False, log_dir=None):
        """配置日志记录器
        
        Args:
            name (str): 日志记录器名称
            debug (bool): 是否启用调试模式
            log_dir (str): 日志目录
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """
        # 获取日志记录器
        logger = logging.getLogger(name)
        
        # 已配置则直接返回
        if logger.handlers:
            return logger
            
        # 设置日志级别
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # 配置控制台输出
        console_handler = UnicodeStreamHandler(stream=sys.stdout, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 配置文件输出
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'logs'
            )
        
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 添加文件处理器
        try:
            log_file = os.path.join(log_dir, '%s.log' % name.lower())
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            sys.stderr.write("无法创建日志文件: %s\n" % str(e))
            
        return logger