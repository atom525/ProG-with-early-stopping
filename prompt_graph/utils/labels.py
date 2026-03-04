"""图标签安全提取，兼容 y 为标量、1维等不同形状"""


def safe_graph_label(y):
    """兼容 graph.y 为标量 tensor(0)、1维 tensor([0]) 等形状，返回 int"""
    if y is None:
        return 0
    if hasattr(y, 'numel') and y.numel() == 0:
        return 0
    return int(y.flatten()[0].item())
