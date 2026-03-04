from .train_logger import train_info

def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
        train_info("{} has {} parameters".format(name, param))
    train_info("Total Parameters: {}".format(total_params))