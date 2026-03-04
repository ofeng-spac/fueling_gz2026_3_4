import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path
from fueling.initialization.get_model import init_model

from loguru import logger
import json, _jsonnet

def main(camera: str = "", light: str = "", cut: str = ""):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'init_model.log')
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:7} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",  # 日志文件达到10MB时轮转
        retention="7 days",  # 保留7天的日志
        encoding="utf-8"
    )
    config_files = [
        "../../data/arm1/config.jsonnet"
    ]
    for config_path in config_files:
        try:
            json_str = _jsonnet.evaluate_file(config_path)
            config = json.loads(json_str)
            print(config.keys(),'\n')
            logger.info("成功加载配置文件: {}", config_path)
            target_pot = config["robot"]["target_pot"]
            logger.info("目标壶: {}", target_pot)
        except Exception as e:
            logger.error("加载配置文件或参数失败: {}", e)
            continue
        pots_config_path = Path(config_path).parent / target_pot / "robot_pose.json"  # 新的
        pots_config_path.parent.mkdir(parents=True, exist_ok=True)
        init_model(light, config,  pots_config_path, camera, cut)
        logger.info("配置文件 {} 初始化模型成功", config_path)
if __name__ == "__main__":
    main(camera="OB", light="flood",  cut="box")


