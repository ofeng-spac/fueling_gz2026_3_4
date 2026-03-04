import numpy as np

txt_path = "output.txt"
save_dir = "result"

import os
os.makedirs(save_dir, exist_ok=True)

with open(txt_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue  # 跳过空行

        # 将字符串行转为数值列表（按空格分隔）
        arr = np.array(list(map(float, line.split())))

        # 保存为 npy
        save_path = os.path.join(save_dir, f"result{i+1}.npy")
        np.save(save_path, arr)

        print(f"Saved: {save_path}")