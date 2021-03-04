
from torchvision.utils import save_image as _save_image
import threading as _threading
import os as _os
from tqdm.auto import tqdm as _tqdm
import sys as _sys
import multiprocessing as _threading


def thread_save_image(tensor,
                      base_dir,
                      file_type:str="jpg",
                      normalize=True,
                      max_threads=32,
                      prefix="",
                      suffix="",
                      base_num=1,
                      placeholder="0",
                      just_length=10,
                      batch_size=2048):
    '''
    Parameters:
    tensor: images shape as : BxCxLxH
    base_dir: base directory of image will be saved
    file_type: save type
    prefix: file name prefix
    suffix: file name suffix
    base_num: file name is a number string and increase from base_num
    just_length: the length of string
    batch_size: how many images should be saved in every process
    '''
    s = len(tensor)
    max_threads = min(64, max_threads)
    if s < batch_size * max_threads:
        batch_size = s // max_threads
    semaphor = _threading.Semaphore(value=max_threads)
    length = s // batch_size + 1
    if s % batch_size == 0:
        length -= 1
    with _tqdm(total=length) as pbar:
        pbar.set_description_str("Saving")
        for idx in range(length):
            if len(str(base_num + idx)) > just_length:
                raise RuntimeError(f"the length of {base_num + idx} > {just_length}")
            semaphor.acquire()
            thread = _threading.Process(target=_thread_save_image,
                                  args=(
                                      tensor[idx * batch_size:(idx + 1) * batch_size],
                                      base_dir,
                                      file_type,
                                      normalize,
                                      prefix,
                                      suffix,
                                      base_num + batch_size * idx,
                                      placeholder,
                                      just_length
                                  ))
            thread.start()
            pbar.update(1)
            semaphor.release()
        thread.join()
        pbar.set_description_str("Finished")
        print("All images will be saved after a few seconds ...", file=_sys.stderr)

def _thread_save_image(tensor,
                      base_dir,
                      file_type:str="jpg",
                      normalize=False,
                      prefix="",
                      suffix="",
                      base_num=0,
                      placeholder="0",
                      just_length=10):
    for idx, image in enumerate(tensor):
        if len(str(base_num + idx)) > just_length:
            raise RuntimeError(f"the length of {base_num + idx} > {just_length}")

        name = prefix + str(base_num + idx).rjust(just_length, placeholder)\
                            + suffix + "." + file_type
        name = _os.path.join(base_dir, name)
        _save_image(image, name, normalize=normalize)
