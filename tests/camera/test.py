import anyio
from obcamera.obcamera import OrbbecCamera
import cv2

async def test():
    # 实例化相机
    obcamera = OrbbecCamera(
        id = 0,
        pipeline_params={
            'enable_streams': [{'type': 'IR'}]
        }
    )

    # 采集一对红外图像
    images = await obcamera.capture_stereo_ir(exposure_val=3000, gain_val=4800, is_flood_on=True)

    # 保存图像
    cv2.imwrite('left_ir_main.png', images['left_ir'])
    cv2.imwrite('right_ir_main.png', images['right_ir'])

    print("红外图像已采集并保存为 left_ir_main.png 和 right_ir_main.png")

async def main():
    await test()
if __name__ == "__main__":
    anyio.run(main)
