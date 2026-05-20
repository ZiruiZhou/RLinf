from .lingbotva_wan_action_model import LingbotvaWanActionModel


def get_model(cfg, torch_dtype):
    return LingbotvaWanActionModel(cfg, torch_dtype)
