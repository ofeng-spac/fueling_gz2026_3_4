
from fueling.obcamera import AsyncOrbbecCamera
import cv2
import asyncio
import logging
async def example_usage():
    logger = logging.getLogger("obcamera_example")
    # 初始化相机，确保配置中包含彩色流
    pipeline_params = {
        'enable_streams': [
            {'type': 'COLOR', 'width': 1920, 'height': 1080, 'fps': 30}
        ]
    }
    
    camera = AsyncOrbbecCamera(id=0, pipeline_params=pipeline_params)
    
    try:
        # 使用默认参数捕获彩色图像
        color_image = await camera.capture_color()
        
        # 或者使用自定义参数
        color_image = await camera.capture_color(
            exposure_val=2000,      # 曝光时间
            gain_val=24,            # 增益
            auto_exposure=False,    # 手动曝光
            timeout_ms=3000         # 超时时间
        )
        
        # 保存或显示图像
        cv2.imwrite('./color_image.jpg', color_image)
        cv2.imshow('Color Image', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Failed to capture color image: {e}")
    
    finally:
        camera.stop()


asyncio.run(example_usage())