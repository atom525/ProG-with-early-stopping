import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        from prompt_graph.utils.train_logger import train_info
        train_info("create folder {}".format(path))
    else:
        from prompt_graph.utils.train_logger import train_info
        train_info("folder exists {}".format(path))
