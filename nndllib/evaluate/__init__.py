
from .nndl_eval import *
import os
from torchvision.utils import make_grid

'''
Date : 01/14/2021
Author : HuChang
'''

__version__ = "1.0.0"
__all__ = [
    "ClassifierEvaluate"
]
__path__ = os.path.abspath(".")
__str__ = os.path.abspath(".")
__repr__ = __str__

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def imshow(data, pause=0.0001, mu=0.5, sigma=0.5):
    data = data.to("cpu")
    images = make_grid(data).data.numpy().transpose(1, 2, 0) * sigma + mu
    plt.imshow(images)
    plt.pause(pause)