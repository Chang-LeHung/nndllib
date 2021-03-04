#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Mr. HuChang

# from .fast_torch import *
#
# __version__ = '1.0.0'


import multiprocessing
import time
# 不能将共享变量和共享锁定义成全局变量然后通过global引用那样会报错，只能传过来
def sub_process(process_name,share_var,share_lock):
    # 获取锁
    print("______________________________________")
    time.sleep(5)
    share_lock.acquire()
    share_var.append(process_name)
    # 释放锁
    share_lock.release()
    for item in share_var:
        print(f"{process_name}-{item}")

def main_process():
    # 单个值声明方式。typecode是进制类型，C写法和Python写法都可以，见下方表格；value是初始值。
    # 这种单值形式取值赋值需要通过get()/set()方法进行，不能直接如一般变量那样取值赋值
    # share_var = multiprocessing.Manager().Value(typecode, value)
    # 数组声明方式。typecode是数组变量中的变量类型，sequence是数组初始值
    # share_var = multiprocessing.Manager().Array(typecode, sequence)
    # 字典声明方式
    # share_var = multiprocessing.Manager().dict()
    # 列表声明方式
    share_var = multiprocessing.Manager().list()
    share_var.append("start flag")
    # 声明一个进程级共享锁
    # 不要给多进程传threading.Lock()或者queue.Queue()等使用线程锁的变量，得用其进程级相对应的类
    # 不然会报如“TypeError: can't pickle _thread.lock objects”之类的报错
    share_lock = multiprocessing.Manager().Lock()
    process_list = []

    process_name = "process 1"
    tmp_process = multiprocessing.Process(target=sub_process,args=(process_name,share_var,share_lock))
    process_list.append(tmp_process)

    process_name = "process 2"
    tmp_process = multiprocessing.Process(target=sub_process, args=(process_name,share_var,share_lock))
    process_list.append(tmp_process)

    process_name = "process 3"
    tmp_process = multiprocessing.Process(target=sub_process, args=(process_name,share_var,share_lock))
    process_list.append(tmp_process)

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    print(share_var, "在主进程")

if __name__ == "__main__":
    main_process()
