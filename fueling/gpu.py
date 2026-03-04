import pynvml

def get_gpu_mem_usage() -> tuple[float, float]:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / 1024**3          # GB
    total = info.total / 1024**3        # GB
    return used, total