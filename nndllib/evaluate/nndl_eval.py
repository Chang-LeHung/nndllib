# %load /usr/local/lib/python3.7/site-packages/hu_chang/evaluate/nndl_eval.py
# %load /usr/local/lib/python3.7/site-packages/hu_chang/evaluate/nndl_eval.py
import torch
from torch import nn
from tqdm.auto import tqdm


class DistributionEvaluate(object):
    
    '''
    用于评估分类后的概率分布，2021/1/15
    默认使用 tqdm_notebook 
    '''

    def __init__(self):
        pass

    @staticmethod
    def softmax_evaluate(model, data_loader, classes=None, call_backend=lambda x : x, softmax=True):
        '''
        model : 待评价的模型
        data_loader  : 待遍历的 loader
        classes : 输入类型为列表 表示有多少类 如为5类 [0, 1, 2, 3, 4]
        softmax : 模型的输出是否需要再经过 softmax, 默认为 True
        return  : 返回两个字典，分别为分类正确和错误的字典，字典内容为
                  每一类的经过softmax的输出，字典的键为labels的内容
                  还有真个 Dataloader 的正确率 一共三个返回值
                
        '''
        correct = dict([ (c, []) for c in classes])
        error   = dict([ (c, []) for c in classes])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        err_rate = 0
        cor_rate = 0
        if softmax is True:
            soft = nn.Softmax(dim=1)
            with tqdm(total=len(data_loader)) as pbar:
                for data, labels, in data_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    out = soft(model(call_backend(data)))
                    idx = out.argmax(dim=1)
                    ans = (idx == labels)
                    for idx, item in enumerate(ans):
                        if item.item() == 1:
                            correct[labels[idx].item()].append(out[idx].data.tolist())
                            cor_rate += 1
                        else:
                            error[labels[idx].item()].append(out[idx].data.tolist())
                            err_rate += 1
                    pbar.update(1)
                    pbar.set_description("Working")
                pbar.set_description("Finshed")
        else:
            with tqdm(total=len(data_loader)) as pbar:
                for data, labels, in data_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    out = model(call_backend(data))
                    idx = out.argmax(dim=1)
                    ans = (idx == labels)
                    for idx, item in enumerate(ans):
                        if item.item() == 1:
                            correct[labels[idx].item()].append(out[idx].data.tolist())
                            cor_rate += 1
                        else:
                            error[labels[idx].item()].append(out[idx].data.tolist())
                            err_rate += 1
                    pbar.update(1)
                    pbar.set_description("Working")
                pbar.set_description("Finshed")
        return dict([(key, torch.Tensor(val)) for key, val in correct.items()]), \
                dict([(key, torch.Tensor(val)) for key, val in error.items()]), (cor_rate) / (cor_rate + err_rate)

    @staticmethod
    def key_soft_exp_normalize(data):
        '''
        data : 输入维度 ：batch_size X dim
                对于一条数据 x1, x2, ..., x_n
                返回的位 y1, y2, y3, ..., y_n

                y_i = exp(x_i) / sum(exp(i) , i : 0 -> n)
        '''
        return dict([(key, DistributionEvaluate.__key_soft(val)) for key, val in data.items()])

    @classmethod
    def __key_soft(cls, data):
        data = data.exp()
        temp = data.sum(dim=1).unsqueeze(1)
        return data / temp

