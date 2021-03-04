import sys as _sys
import seaborn as _sns
from torch import nn as _nn
import torch as _torch
from torchvision.utils import make_grid as _mg
from matplotlib import pyplot as _plt
import numpy as _np
from tqdm.auto import tqdm as _tqdm
from .multi_save_image import thread_save_image
import multiprocessing as _processes


_sns.set()


class GANShower(object):
    '''
    iters : 每一个 epoch 的迭代次数
    start : 每一次迭代的起始数 默认从 0 开始
    show_log : 是否打印日志
    show_image : 是否展示生成的图片
    show_loss_curve : 是否显示 loss 的曲线
    log_interval : 打印日志的迭代次数
    image_interval : 显示图片的迭代次数
    loss_curve_interval : 打印损失曲线的 epoch 间隔
    pause : 显示图片的时间 默认为 0.0001
    file : 日志输出的文件 默认为 sys.stdout
    
    Example : 
    >>> import torch
    >>> from hu_chang.utils import GANShower
    >>> er = GANShower(1000, image_interval=300, log_interval=18, show_loss_curve=False)
    >>> for i in range(3):
    >>>     for j in range(1000):
    >>>         er.add(torch.rand(1), torch.rand(1))
    >>>         log, img = er.check()
    >>>         if img : 
    >>>             er.imshow(torch.rand(16, 3, 28, 28))
    '''
    def __init__(self, 
                 iters,
                 start=0,
                 show_log=True, 
                 show_loss_curve=True,
                 show_image=True, 
                 log_interval=10, 
                 image_interval=10,
                 loss_curve_interval=1,
                 pause=0.0001,
                 file=_sys.stdout
                ):
        
        self.iters = iters
        self.start = start
        self.show_log = show_log
        self.show_image = show_image
        self.show_loss_curve = show_loss_curve
        self.log_interval = log_interval
        self.image_interval = image_interval
        self.loss_curve_interval = loss_curve_interval
        self.pause = pause
        self.file = file
        
        self.trained_epochs = 0
        self.dis_avg_loss_per_epoch = []
        self.gen_avg_loss_per_epoch = []
        self.cur_iter = start
        self._dis_temp_loss = []
        self._gen_temp_loss = []

    def add(self, dis, gen):
        '''
        dis : 为判别器的损失 torch.Size([1]) 的 Tensor
        gen : 为生成器的损失 torch.Size([1]) 的 Tensor
        '''
        self._dis_temp_loss.append(dis.item())
        self._gen_temp_loss.append(gen.item())
        self.cur_iter += 1
        log, img = self._check()

        if log : 
            print(f"Epoch = {self.trained_epochs + 1} step[{self.cur_iter}/{self.iters}]" \
                f"dis loss = {dis.item()} gen loss = {gen.item()}", file=self.file)
        self._update()

    def _update(self):
        
        if self.cur_iter == self.iters:
            self.trained_epochs += 1
            self.cur_iter = 0
            self.dis_avg_loss_per_epoch.append(_np.mean(self._dis_temp_loss))
            self.gen_avg_loss_per_epoch.append(_np.mean(self._gen_temp_loss))
            self._dis_temp_loss = []
            self._gen_temp_loss = []
            if self.trained_epochs % self.loss_curve_interval == 0 and self.show_loss_curve: 
                _sns.lineplot(x = _np.arange(self.trained_epochs), y = self.dis_avg_loss_per_epoch, 
                                                                          label="discriminator")
                _sns.lineplot(x = _np.arange(self.trained_epochs), y = self.gen_avg_loss_per_epoch, 
                                                                          label="generator")
                _plt.pause(self.pause)

    def check(self):
        
        log = False
        img = False
        if self.cur_iter % self.image_interval == 0 and self.show_image:
            img = True
            
        if self.cur_iter % self.log_interval == 0 and self.show_log:
            log = True
            
        return log, img

    def _check(self):
        
        log = False
        img = False
        if self.cur_iter % self.image_interval == 0 :
            img = True
            
        if self.cur_iter % self.log_interval == 0:
            log = True

        return log, img
    
    def imshow(self, data):
        data = data.to("cpu")
        images = _mg(data).data.numpy().transpose(1, 2, 0) * 0.5 + 0.5
        _plt.imshow(images)
        _plt.pause(self.pause)


class MyDataset(_nn.Module):
    
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self, idx):
        
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class BaseShower(object):
    '''
    Args:
    
    iters : 每一个 epoch 的迭代次数
    show_log : 是否打印日志
    show_loss_curve : 是否显示 loss 的曲线
    log_interval : 打印日志的迭代次数
    loss_curve_interval : 打印损失曲线的 epoch 间隔
    pause : 显示图片的时间 默认为 0.0001
    file : 日志输出的文件 默认为 sys.stdout
    
    Example : 
    >>> import torch
    >>> from hu_chang.utils import BaseShower
    >>> shower = BaseShower(500)
    >>> for i in range(10):
    >>>     for j in range(500):
    >>>         shower.add(torch.rand(1))
    '''
    def __init__(self, 
                 iters,
                 show_log=True, 
                 show_loss_curve=True,
                 log_interval=10, 
                 loss_curve_interval=1,
                 pause=0.0001,
                 file=_sys.stdout
                ):
        
        self.iters = iters
        self.trained_epochs = 0
        self.cur_iter = 0
        self.show_log = show_log
        self.show_loss_curve = show_loss_curve
        self.log_interval = log_interval
        self.loss_curve_interval = loss_curve_interval
        self.pause = pause
        self.avg_loss_per_epoch = []
        self._temp_loss =[]
        self.file = file
        
    def add(self, loss):
        '''
        loss : 损失 torch.Size([1]) 的 Tensor
        '''
        self._temp_loss.append(loss.item())
        self.cur_iter += 1
        log = self._check()

        if log : 
            print(f"Epoch = {self.trained_epochs + 1} step[{self.cur_iter}/{self.iters}]" \
                f"loss = {loss.item()}", file=self.file)
        self._update()

    def _update(self):
        
        if self.cur_iter == self.iters:
            self.trained_epochs += 1
            self.cur_iter = 0
            self.avg_loss_per_epoch.append(_np.mean(self._temp_loss))
            self._temp_loss = []
            if self.trained_epochs % self.loss_curve_interval == 0 and self.show_loss_curve: 
                ax = _sns.lineplot(x = _np.arange(self.trained_epochs), y = self.avg_loss_per_epoch, 
                                                                          label="loss curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                _plt.pause(self.pause)

    def check(self):
        
        log = False
        if self.cur_iter % self.log_interval == 0 and self.show_log:
            log = True
        return log

    def _check(self):
        
        log = False
        if self.cur_iter % self.log_interval == 0:
            log = True
        return log


class ClfShower(BaseShower):
    '''
    Args:
    iters : 每一个 epoch 的迭代次数
    show_log : 是否打印日志
    show_loss_curve : 是否显示 loss 的曲线
    log_interval : 打印日志的迭代次数
    loss_curve_interval : 打印损失曲线的 epoch 间隔
    pause : 显示图片的时间 默认为 0.0001
    file : 日志输出的文件 默认为 sys.stdout
    same as BaseShower
    
    Example : 
    >>> import torch
    >>> from hu_chang.utils import BaseShower
    >>> shower = BaseShower(500)
    >>> for i in range(10):
    >>>     for j in range(500):
    >>>         acc = 0
    >>>         shower.add(torch.rand(1), acc)
    '''
    def __init__(self, 
                 iters,
                 show_log=True, 
                 show_loss_curve=True,
                 log_interval=10, 
                 loss_curve_interval=1,
                 pause=0.0001,
                 file=_sys.stdout
                ):
        
        super(ClfShower, self).__init__(
                iters,
                show_log,
                show_loss_curve,
                log_interval,
                loss_curve_interval,
                pause,
                file
        )

    def add(self, loss, acc=None, val_acc=None):
        '''
        loss : 损失 torch.Size([1]) 的 Tensor
        '''
        self._temp_loss.append(loss.item())
        self.cur_iter += 1
        log = self._check()
        if log :
            print(f"Epoch = {self.trained_epochs + 1} step[{self.cur_iter}/{self.iters}]" \
                        f"loss = {loss.item()}", file=self.file, end=" ")
            if acc is not None:
                print(f"acc={acc}", file=self.file, end=" ")
            if val_acc is not None:
                print(f"val_acc={val_acc}", file=self.file, end = " ")
            print()
        self._update()


class ClfShower(BaseShower):
    '''
    Args:
    iters : 每一个 epoch 的迭代次数
    show_log : 是否打印日志
    show_loss_curve : 是否显示 loss 的曲线
    log_interval : 打印日志的迭代次数
    loss_curve_interval : 打印损失曲线的 epoch 间隔
    pause : 显示图片的时间 默认为 0.0001
    file : 日志输出的文件 默认为 sys.stdout
    same as BaseShower
    
    Example : 
    >>> import torch
    >>> from hu_chang.utils import BaseShower
    >>> shower = BaseShower(500)
    >>> for i in range(10):
    >>>     for j in range(500):
    >>>         acc = 0
    >>>         shower.add(torch.rand(1), acc)
    '''
    def __init__(self, 
                 iters,
                 show_log=True, 
                 show_loss_curve=True,
                 log_interval=10, 
                 loss_curve_interval=1,
                 pause=0.0001,
                 file=_sys.stdout
                ):
        
        super(ClfShower, self).__init__(
                iters,
                show_log,
                show_loss_curve,
                log_interval,
                loss_curve_interval,
                pause,
                file
        )

    def add(self, loss, acc=None, val_acc=None):
        '''
        loss : 损失 torch.Size([1]) 的 Tensor
        '''
        self._temp_loss.append(loss.item())
        self.cur_iter += 1
        log = self._check()
        if log :
            print(f"Epoch = {self.trained_epochs + 1} step[{self.cur_iter}/{self.iters}]" \
                        f"loss = {loss.item()}", file=self.file, end=" ")
            if acc is not None:
                print(f"acc={acc}", file=self.file, end=" ")
            if val_acc is not None:
                print(f"val_acc={val_acc}", file=self.file, end = " ")
            print()
        self._update()


class ClfPipe(ClfShower):
    '''
    该类是用于分类模型中进行训练和评估，类似与 fit 的作用
    Args : 
    epochs : 训练的epoch的数目
    model : 待训练的模型
    criterion : 损失函数模型
    optimizer : model的优化器
    train_loader : 训练集的 Dataloader 进行一次迭代返回的是 data, labels
    cal_acc : 对于每一个迭代产生的 data 是否计算 其正确率
    cal_test ： 是否对于整个测试集是否进行计算正确率
    val_acc_interval : 默认为 100
    train_acc_interval : 计算正确率的间隔
    test_loader : 测试集的 DataLoader
    show_log : 是否打印日志
    show_loss_curve : 是否显示 loss 的曲线
    log_interval : 打印日志的迭代次数
    loss_curve_interval : 打印损失曲线的 epoch 间隔
    pause : 显示图片的时间 默认为 0.0001
    file : 日志输出的文件 默认为 sys.stdout
    注意：val_acc_interval train_acc_interval 应该为 log_interval 整数倍 否则不能打印准确率
    Example
    >>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    >>> criterion = nn.CrossEntropyLoss()
    >>> net = CNN().to(device)
    >>> optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0001)
    >>> clf = ClfPipe(epochs=20, model=net, criterion=criterion, optimizer=optimizer, train_loader=trainloader,
             test_loader=testloader, log_interval=1)
    >>> clf.start()
    >>> # To continue train 5 epochs
    >>> clf.train(5)
    
    '''
    def __init__(self, 
                epochs, 
                model, 
                criterion, 
                optimizer, 
                train_loader, 
                cal_acc=True,
                cal_test=True,
                train_acc_interval=10,
                val_acc_interval=100,
                test_loader=None,
                show_log=True,
                show_loss_curve=True,
                show_test_acc_curve=True,
                log_interval=10,
                loss_curve_interval=1,
                pause=0.0001,
                file=_sys.stdout,
                ):
        super(ClfPipe, self).__init__(len(train_loader), show_log=show_log, 
                                      log_interval=log_interval,
                                      loss_curve_interval=loss_curve_interval,
                                      show_loss_curve=show_loss_curve, file=file, pause=pause)
        self.device = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
        self.epochs = epochs
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.acc_per_epoch = []
        self.train_acc_interval = train_acc_interval
        self.val_acc_interval = val_acc_interval
        self.cal_acc = cal_acc
        self.cal_test = cal_test
        self.test_accs = []
        self.show_test_acc_curve = show_test_acc_curve
        self.pause = pause

    def start(self):
        
        self.train(self.epochs)

    def step(self, batch_data):
        '''
        batch_data : next(iter(self.train_loader)) 的输出
        return : 模型的输出和标签
        '''
        data, labels = batch_data
        data = data.to(self.device)
        labels = labels.to(self.device)
        out = self.model(data)
        return out, labels

    def eval(self, train=False):
        self.model.eval()
        loader = self.train_loader
        if not train:
            assert self.test_loader is not None, "test_loader is None , "\
                "please pass test_loader first : clf_eval.test_loader = test_loader"
            loader = self.test_loader
        s = 0
        a = 0
        with _tqdm(total=len(loader)) as pbar:
            pbar.set_description("Evaluating")
            for data in loader:
                pbar.update(1)
                out, labels = self.step(batch_data = data)
                t_a, t_s = self.computer_acc(out, labels)
                a += t_a
                s += t_s
            pbar.set_description("Evaluated")
        self.model.train()
        return a / s

    def computer_acc(self, out, labels):
        assert not isinstance(labels, _torch.LongTensor), "labels 数据类型不为 LongTensor Tensor.long进行转换"
        return (out.argmax(dim=-1) == labels).sum().item(), len(labels)

    def train(self, epochs):
        for _ in range(epochs):
            for data in self.train_loader:
                out, labels = self.step(data)
                test_acc = None
                train_acc = None
                loss = self.criterion(out, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.cal_acc and self.cur_iter % self.train_acc_interval == 0:
                    a, s = self.computer_acc(out, labels)
                    train_acc = a / s
                if self.cal_test and self.cur_iter % self.val_acc_interval == 0:
                    test_acc = self.eval()
                    self.test_accs.append(test_acc)
                    if self.show_test_acc_curve and len(self.test_accs) > 1:
                        ax = _sns.lineplot(x = _np.arange(len(self.test_accs)), y = self.test_accs,
                                          label="Accuarcy on Testset")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        _plt.pause(self.pause)
                self.add(loss, train_acc, test_acc)

class Saver(object):
    
    def __init__(self, deepcopy_model):
        self.best_condition = None
        self.start = False
        self.best_model = deepcopy_model

    def update(self, condition, model, less=True):
        
        if not self.start:
            self.start = True
            self.best_condition = condition
            self.best_model.load_state_dict(model.state_dict())
        else:
            if (less and self.best_condition > condition) or \
                    (not lesss and self.best_condition < condition):
                self.best_condition = condition
                self.best_model.load_state_dict(model.state_dict())

    def save(self, path):
        _torch.save(self.best_model.load_state_dict(), path)

class HingeLoss(_nn.Module):
    
    '''
    Example :
    >>> data = torch.rand(64)
    >>> m = HingleLoss()
    >>> m(data, True) # tensor(0.4407)
    '''
    
    def forward(self, x, positive=True):
        batch_size = x.shape[0]
        loss = None
        if positive :
            loss = _torch.clamp(1 - x, 0)
        else:
            loss = _torch.clamp(1 + x, 0)
        return loss.sum() / batch_size


def _batch_image_generator(generator,
                          noise_z,
                          num_batch=30,
                          batch_size=256,
                          conditional=False,
                          classes=None):
    '''
    Args:
        num_batch : the best num for this function if you need more
                    you can call this function more time for faster speed
    '''
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    data = _torch.randn(batch_size, noise_z).to(device)
    labels = _torch.empty(batch_size).random_(classes).long().to(device)
    generator = generator.to(device).eval()
    with _torch.no_grad():
        if conditional:
            images = generator(data.normal_(), labels).cpu()
        else:
            images = generator(_torch.randn(batch_size, noise_z).to((device))).cpu()
        for i in range(num_batch - 1):

            if conditional:
                imgs = generator(data.normal_(), labels.random_(classes)).cpu()
            else:
                imgs= generator(data.normal_()).cpu()
            images = _torch.cat((images, imgs), dim=0)
            del imgs
            _torch.cuda.empty_cache()
    generator.train()
    return images

def batch_image_generator(generator,
                          noise_z,
                          num_batch=50,
                          batch_size=256,
                          conditional=False,
                          classes=None, best_size=30):
    with _tqdm(total=num_batch) as pbar:
        pbar.set_description_str("Saving images")
        if best_size > num_batch:
            best_size = num_batch
        results = _batch_image_generator(generator, noise_z, best_size, batch_size,
                               conditional, classes)
        pbar.update(best_size)
        num_batch -= best_size
        while num_batch >= best_size:
            out = _batch_image_generator(generator, noise_z, best_size, batch_size,
                                   conditional, classes)
            results = _torch.cat((results, out), dim=0)
            pbar.update(best_size)
            num_batch -= best_size
        if num_batch > 0:
            out = _batch_image_generator(generator, noise_z, num_batch, batch_size,
                                   conditional, classes)
            results = _torch.cat((results, out), dim=0)
            pbar.update(num_batch)
        pbar.set_description_str("Finished")
    return results


def extract_from_dataset(dataset, index=None, topK=None):
    '''
    Args:
        dataet : 代提取的数据集
        idx : 提取 dataset 返回数据的第几个 如果只有一个数据则不填
    '''
    s = len(dataset)
    if topK is not None:
        s = topK
    with _tqdm(total=s) as pbar:
        pbar.set_description_str("Loading images")
        for idx, data in enumerate(dataset):
            if idx == 0:
                if index is None:
                    result = data
                else:
                    result = data[index]
            else:
                if index is None:
                    result = _torch.cat((result, data), dim=0)
                else:
                    result = _torch.cat((result, data[index]), dim=0)
            pbar.update(1)
            if topK is not None and topK == idx + 1:
                break
        pbar.set_description_str("Finished")
        return result


def _normalize_data(tensor):

    from torchvision.utils import make_grid
    for idx in range(len(tensor)):
        tensor[idx].data.copy_(make_grid(tensor[idx], normalize=True))


def images_normalize(tensor, max_process=32):
    delta = len(tensor) // max_process + 1
    processes = []
    for idx in range(max_process):
        data = tensor[idx * delta:(idx + 1) * delta]
        p = _processes.Process(target=_normalize_data,
                           args=(data, ))
        processes.append(p)
    for p in processes:
        p.start()
    print(f"{max_process} processes have been started", file=_sys.stderr)
    with _tqdm(total=len(processes)) as pbar:
        pbar.set_description_str("Executing")
        for p in processes:
            p.join()
            pbar.update(1)
        pbar.set_description_str("Executed")
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', _torch.uint8).numpy()


def _interpolate(tensor, cache, size, scale_factor, mode, align):
    import torch.nn.functional as F
    cache.data.copy_(F.interpolate(
        tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align
    ))

def multiprocess_interpolate(input,
                             caches,
                             max_process=32,
                             size=None,
                             scale_factor=None,
                             mode='bilinear',
                             align_corners=False):

    delta = len(input) // max_process + 1
    processes = []
    for idx in range(max_process):
        data = input[idx * delta:(idx + 1) * delta]
        cache = caches[idx * delta:(idx + 1) * delta]
        p = _processes.Process(target=_interpolate,
                           args=(data, cache, size, scale_factor,
                                 mode, align_corners))
        processes.append(p)
    for p in processes:
        p.start()
    print(f"{max_process} processes have been started", file=_sys.stderr)
    with _tqdm(total=len(processes)) as pbar:
        pbar.set_description_str("Executing")
        for p in processes:
            p.join()
            pbar.update(1)
        pbar.set_description_str("Executed")
    print("All the images have benn interpolated in caches", file=_sys.stderr)
