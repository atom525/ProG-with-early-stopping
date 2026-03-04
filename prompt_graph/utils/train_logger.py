"""
统一训练日志格式（RecBole 风格）
- 带时间戳的 INFO 前缀
- epoch training / evaluating 标准格式
- valid result / best valid / test result 的 OrderedDict 格式
- 每行独立输出，不覆盖，log 同步
"""
import sys
from datetime import datetime
from collections import OrderedDict
import torch


def _timestamp():
    """返回 RecBole 风格时间戳，如 06 Sep 17:13"""
    now = datetime.now()
    return now.strftime("%d %b %H:%M")


def train_info(msg):
    """标准 INFO 输出：timestamp INFO message，每行独立不覆盖"""
    line = "{} INFO {}".format(_timestamp(), msg)
    print(line)
    sys.stdout.flush()


def epoch_training(epoch, time_s, train_loss):
    """epoch N training [time: Xs, train loss: Y]"""
    train_info("epoch {} training [time: {:.2f}s, train loss: {:.6f}]".format(epoch, time_s, train_loss))


def epoch_evaluating(epoch, time_s, valid_score, score_name="valid_score"):
    """epoch N evaluating [time: Xs, valid_score: Y]"""
    train_info("epoch {} evaluating [time: {:.2f}s, {}: {:.6f}]".format(epoch, time_s, score_name, valid_score))


def valid_result(metrics, nonzero_only=True):
    """
    valid result: metric1: val1 metric2: val2 ...
    仅打印非 0 值（nonzero_only=True），便于预训练/下游不同指标混排时简洁输出
    """
    if nonzero_only:
        filtered = {k: v for k, v in metrics.items() if v is not None and (not isinstance(v, (int, float)) or abs(float(v)) > 1e-9)}
    else:
        filtered = dict(metrics)
    if not filtered:
        train_info("valid result: (no nonzero metrics)")
        return
    parts = " ".join("{}: {:.6f}".format(k, v) for k, v in filtered.items())
    train_info("valid result: {}".format(parts))


def best_valid_ordered(metrics, nonzero_only=True):
    """best valid: 仅打印非 0 值"""
    if not isinstance(metrics, (dict, OrderedDict)):
        metrics = OrderedDict([("valid_acc", metrics)])
    metrics = OrderedDict(metrics) if isinstance(metrics, dict) else OrderedDict(metrics)
    if nonzero_only:
        metrics = OrderedDict((k, v) for k, v in metrics.items() if v is not None and (not isinstance(v, (int, float)) or abs(float(v)) > 1e-9))
    train_info("best valid: {}".format(metrics if metrics else "(no nonzero metrics)"))


def test_result_ordered(metrics):
    """test result: OrderedDict([...])"""
    if not isinstance(metrics, OrderedDict):
        metrics = OrderedDict(metrics)
    train_info("test result: {}".format(metrics))


def finished_training(best_epoch):
    """Finished training, best eval result in epoch N"""
    train_info("Finished training, best eval result in epoch {}".format(best_epoch))


def early_stopping_msg(epoch, patience, metric_name='valid_acc'):
    """Early stopping 提示：显示触发的指标名"""
    train_info("Early stopping at epoch {}! ({} not improved for {} evals)".format(epoch, metric_name, patience))


def model_saved(path):
    """模型保存成功"""
    train_info("Model saved to {}".format(path))


def model_loaded(path):
    """模型加载"""
    train_info("Loading model structure and parameters from {}".format(path))


def log_args(args, prefix="args"):
    """训练启动时记录完整超参数（便于复现）"""
    if args is None:
        return
    try:
        d = vars(args)
    except TypeError:
        d = dict(args) if hasattr(args, 'items') else {}
    train_info("=" * 50 + " " + prefix)
    for k in sorted(d.keys()):
        v = d[k]
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        train_info("  {}: {}".format(k, v))
    train_info("=" * 50)


def to_ordered_metrics(acc, f1, roc, prc, prefix="valid"):
    """将 acc,f1,roc,prc 转为 OrderedDict"""
    return OrderedDict([
        (f"{prefix}_acc", acc),
        (f"{prefix}_f1", f1),
        (f"{prefix}_auroc", roc),
        (f"{prefix}_auprc", prc),
    ])

# 统一指标名，早停参考其一，其余全部打印
VALID_METRIC_KEYS = ['valid_acc', 'valid_f1', 'valid_auroc', 'valid_auprc']

def metric_from_dict(metrics, metric_name='valid_acc'):
    """从 metrics 中取出早停参考的指标值"""
    m = metric_name or 'valid_acc'
    return metrics.get(m, metrics.get('valid_acc', 0.0))


def get_gpu_memory_str():
    """返回 GPU 显存字符串，如 '0.72 G/39.50 G'"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return "{:.2f} G/{:.2f} G".format(used, total)
    return "0.00 G/0.00 G"
