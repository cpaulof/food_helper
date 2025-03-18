from .mobilenet_pretrained import MobilenetModel 
from .custom1 import CustomModel1
from .custom_vit_based import CustomViT



models = {
    "mobilenet": MobilenetModel,
    "custom1": CustomModel1,
    "custom_vit_based": CustomViT
}