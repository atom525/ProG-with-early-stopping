"""
训练日志：同时输出到控制台和文件
"""
import os
import sys
from datetime import datetime


class TeeLogger:
    """将 print 同时写入控制台和 log 文件"""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        dir_ = os.path.dirname(log_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        self.log = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def setup_logging(log_path):
    """设置 TeeLogger，返回 logger 实例"""
    if log_path:
        logger = TeeLogger(log_path)
        sys.stdout = logger
        return logger
    return None
