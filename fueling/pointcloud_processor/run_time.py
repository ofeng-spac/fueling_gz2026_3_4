import time
from functools import wraps
from loguru import logger

def timeit(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.perf_counter()    # 高精度计时
        result = func(*args,**kwargs) # 执行原函数
        end = time.perf_counter()
        logger.info(f"{func.__name__} 耗时: {end - start:.4f}秒")
        return result
    return wrapper

