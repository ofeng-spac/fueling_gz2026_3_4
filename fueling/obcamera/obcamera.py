import pyorbbecsdk as pyob
import time
import asyncio
import numpy as np
import cv2
from pyorbbecsdk import OBSensorType, OBFormat, OBPropertyID, OBFrameType
from ..error import IRImageDecodeError
from loguru import logger
from typing import Optional, Literal


OBPropertyID = pyob.OBPropertyID
CHANNEL_IR_LEFT = 0
CHANNEL_IR_RIGHT = 1

def list_connected_cameras():
    """
    List all connected Orbbec cameras.
    """
    ctx = pyob.Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        logger.error("No Orbbec cameras connected.")
        return
    logger.info("Connected Orbbec cameras:")
    for i in range(device_list.get_count()):
        device = device_list.get_device_by_index(i)
        device_info = device.get_device_info()
        logger.info(f"  - Index: {i}, Serial Number: {device_info.get_serial_number()}")

def find_device(id: Optional[str | int] = None, context: Optional[pyob.Context] = None):
    if context is None:
        context = pyob.Context()
    # if context is not None:
    device_list = context.query_devices() # type: ignore
    if len(device_list) == 0:
        raise RuntimeError("Device not found")

    if type(id) is str:
        serial_number = id
        idx = None
    elif type(id) is int:
        serial_number = None
        idx = id
    else:
        # Default to first device if id is None
        serial_number = None
        idx = 0

    if serial_number is not None:
        selected_device = None
        for device in device_list:
            if device.get_serial_number() == serial_number:
                selected_device = device
                break
        if selected_device is None:
            raise ValueError(f"Serial Number {serial_number} not found")
    else:
        if idx is not None:
            if idx >= len(device_list):
                logger.error(f"Device index {idx} out of range, using default device 0")
                raise ValueError(f"Device index {idx} out of range")

        selected_device = device_list[idx]

    return selected_device

def set_camera_params(dev: pyob.Device, cam_params: dict[str, int | float | bool]):
    
    for key, v in cam_params.items():
        key_id = getattr(OBPropertyID, key)
        param_type = key.split('_')[-1].lower()
        if param_type == 'int':
            dev.set_int_property(key_id, v)
            if dev.get_int_property(key_id) != v:
                logger.warning(f"Failed to set int property {key} to {v}")
        elif param_type == 'bool':
            dev.set_bool_property(key_id, v)
            if dev.get_bool_property(key_id) != v:
                logger.warning(f"Failed to set bool property {key} to {v}")
        elif param_type == 'float':
            dev.set_float_property(key_id, v)
            if dev.get_float_property(key_id) != v:
                logger.warning(f"Failed to set float property {key} to {v}")
def get_camera_params(dev: pyob.Device, cam_params: list[str]):
    params = {}
    for key in cam_params:
        key_id = getattr(OBPropertyID, key)
        param_type = key.split('_')[-1].lower()
        if param_type == 'int':
            params[key] = dev.get_int_property(key_id)
        elif param_type == 'bool':
            params[key] = dev.get_bool_property(key_id)
        elif param_type == 'float':
            params[key] = dev.get_float_property(key_id)
    return params
def set_pipeline_config(params: dict, config: Optional[pyob.Config] = None):
    """Set pipeline configuration based on parameters"""
    if config is None:
        config = pyob.Config()
    if config is not None:
        if 'enable_streams' in params:
            for stream in params['enable_streams']:
                # Map stream types to OBSensorType
                if stream['type'] == 'IR' or stream['type'] == 'OB_STREAM_IR':
                    config.enable_video_stream(OBSensorType.IR_SENSOR)
                elif stream['type'] == 'DEPTH' or stream['type'] == 'OB_STREAM_DEPTH':
                    config.enable_video_stream(OBSensorType.DEPTH_SENSOR)
                elif stream['type'] == 'COLOR' or stream['type'] == 'OB_STREAM_COLOR':
                    config.enable_video_stream(OBSensorType.COLOR_SENSOR)
                else:
                    logger.warning(f"Unknown stream type: {stream['type']}")
    return config

def convert_frame(frame: pyob.Frame):
    """
    Process a frame (depth, IR, or color) and convert it to a displayable BGR image.
    """
    if frame is None:
        return None

    image = None
    frame_type = frame.get_type()

    if frame_type == OBFrameType.DEPTH_FRAME:
        depth_frame = frame.as_depth_frame()
        depth_data = np.asanyarray(depth_frame.get_data())
        width = depth_frame.get_width()
        height = depth_frame.get_height()

        depth_data = np.frombuffer(depth_data, dtype=np.uint16).reshape(height, width)
        
        # Normalize for visualization
        depth_image_8u = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
        image = cv2.applyColorMap(depth_image_8u, cv2.COLORMAP_JET)

    elif frame_type == OBFrameType.IR_FRAME:
        ir_frame = frame.as_ir_frame()
        ir_data = np.asanyarray(ir_frame.get_data())
        width = ir_frame.get_width()
        height = ir_frame.get_height()
        ir_format = ir_frame.get_format()

        if ir_format == OBFormat.Y8:
            ir_data = np.resize(ir_data, (height, width))
        elif ir_format == OBFormat.MJPG:
            ir_data = cv2.imdecode(ir_data, cv2.IMREAD_GRAYSCALE)
            if ir_data is None:
                raise IRImageDecodeError(ir_format, "MJPG decoding returned None")
        elif ir_format == OBFormat.Y16:
            ir_data = np.frombuffer(ir_data, dtype=np.uint16).reshape(height, width)
        else: # Default to Y16 for other cases
            logger.warning(f"Unsupported IR format {ir_format}, trying to decode as Y16")
            ir_data = np.frombuffer(ir_data, dtype=np.uint16).reshape(height, width)

        # Normalize for visualization
        ir_image_8u = cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
        image = cv2.cvtColor(ir_image_8u, cv2.COLOR_GRAY2BGR)

    elif frame_type == OBFrameType.COLOR_FRAME:
        color_frame = frame.as_color_frame()
        color_data = np.asanyarray(color_frame.get_data())
        width = color_frame.get_width()
        height = color_frame.get_height()
        color_format = color_frame.get_format()

        if color_format == OBFormat.RGB:
            image = cv2.cvtColor(color_data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            image = color_data.reshape(height, width, 3)
        elif color_format == OBFormat.YUYV:
            image = cv2.cvtColor(color_data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        else:
            logger.warning(f"Unsupported color format: {color_format}")
            return None
            
    return image

def convert_frameset(frame_set: pyob.FrameSet):
    """处理帧集合，转换为numpy数组字典"""
    frame_dict = {}

    color_frame = frame_set.get_frame(OBFrameType.COLOR_FRAME)
    if color_frame is not None:
        frame_dict['color'] = convert_frame(color_frame)

    depth_frame = frame_set.get_frame(OBFrameType.DEPTH_FRAME)
    if depth_frame is not None:
        frame_dict['depth'] = convert_frame(depth_frame)

    ir_frame = frame_set.get_frame(OBFrameType.IR_FRAME)
    if ir_frame is not None:
        frame_dict['ir'] = convert_frame(ir_frame)

    left_ir_frame = frame_set.get_frame(OBFrameType.LEFT_IR_FRAME)
    if left_ir_frame is not None:
        frame_dict['left_ir'] = convert_frame(left_ir_frame)

    right_ir_frame = frame_set.get_frame(OBFrameType.RIGHT_IR_FRAME)
    if right_ir_frame is not None:
        frame_dict['right_ir'] = convert_frame(right_ir_frame)

    return frame_dict
class AsyncOrbbecCamera:
    def __init__(self, camera_serial: str, pipeline_params: dict = {}, fps: int = 30):
        """
        初始化异步Orbbec相机
        
        Args:
            camera_serial: 相机序列号
            pipeline_params: 管道参数
            fps: 帧率
        """
        self.context = pyob.Context()
        self.camera_serial = camera_serial
        self.device = None
        self.pipeline = None
        self.dev = None
        self.config = pyob.Config()
        self.interval = 1000 // fps
        
        # 先查找设备
        self.find_device()
        
        # 初始化pipeline
        self.pipeline = pyob.Pipeline(self.device)
        self.dev = self.pipeline.get_device()
        
        # 设置管道配置
        set_pipeline_config(pipeline_params, config=self.config)

    def find_device(self):
        """根据序列号查找设备"""
        device_list = self.context.query_devices()
        if len(device_list) == 0:
            raise RuntimeError("No Orbbec devices found")

        # 根据序列号查找设备
        selected_device = None
        for device in device_list:
            device_info = device.get_device_info()
            if device_info.get_serial_number() == self.camera_serial:
                selected_device = device
                break
        
        if selected_device is None:
            available_serials = [device.get_device_info().get_serial_number() for device in device_list]
            raise ValueError(f"Camera with serial {self.camera_serial} not found. Available cameras: {available_serials}")
        
        self.device = selected_device
        device_info = self.device.get_device_info()
        logger.info(f"Camera initialized: Serial: {device_info.get_serial_number()}, Name: {device_info.get_name()}")
        return self.device

    def flush_cache(self):
        """Clear camera cache by draining the frame buffer."""
        try:
            count = 0
            while self.pipeline.wait_for_frames(10) is not None:  # 10ms timeout
                count += 1
            if count > 0:
                logger.debug(f"Flushed {count} cached frames")
        except Exception as e:
            logger.warning(f"Error flushing cache: {e}")

    async def start(self):
        """启动相机流"""
        try:
            self.pipeline.start(self.config)
            await asyncio.sleep(0.2)  # 增加启动延迟确保稳定
            logger.debug("Camera started")
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise

    def stop(self):
        try:
            self.pipeline.stop()
            logger.debug("Camera stopped")
        except Exception as e:
            logger.warning(f"Error stopping camera: {e}")

    async def capture_frames(self, timeout_ms: int = 3000):
        await self.start()
        self.flush_cache()
        frames = None
        start_t = time.time()
        
        logger.debug("Starting frame capture")
        
        capture_attempts = 0
        while frames is None:
            try:
                frames = self.pipeline.wait_for_frames(100)  # 100ms timeout
                capture_attempts += 1
                
                if frames is not None:
                    break
                    
                await asyncio.sleep(0.01)
                
                if time.time() - start_t > timeout_ms / 1000:
                    logger.warning(f"Frame capture timeout after {timeout_ms}ms")
                    break
                    
            except Exception as e:
                logger.error(f"Error during frame capture: {e}")
                await asyncio.sleep(0.1)
        
        self.stop()

        if frames is None:
            raise RuntimeError(f"Failed to capture frame within {timeout_ms}ms")

        logger.debug(f"Successfully captured frame after {capture_attempts} attempts")
        return convert_frameset(frames)

    async def capture_ir(self, 
                         is_right: bool = False, 
                         exposure_val: int = 2000, 
                         gain_val: int = 16, 
                         light_mode: Literal['flood', 'laser', 'none'] = 'flood'):
        if is_right:
            chn = CHANNEL_IR_RIGHT
            side = "right"
        else:
            chn = CHANNEL_IR_LEFT
            side = "left"

        logger.info(f"Capturing {side} IR")
        
        try:
            set_camera_params(self.dev, {'OB_PROP_IR_CHANNEL_DATA_SOURCE_INT': chn})
            set_camera_params(self.dev, {'OB_PROP_IR_EXPOSURE_INT': exposure_val})
            set_camera_params(self.dev, {'OB_PROP_IR_GAIN_INT': gain_val})
            
            # 设置光源
            if light_mode == 'flood':
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': True})
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': False})
            elif light_mode == 'laser':
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': False})
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': True})
            elif light_mode == 'none':
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': False})
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': False})
            else:
                logger.warning(f"Unknown light mode: {light_mode}, using flood")
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': True})
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': False})

            frames = await self.capture_frames()
            
            # 优先使用左右IR帧，如果没有则使用普通IR帧
            if is_right and 'right_ir' in frames and frames['right_ir'] is not None:
                ir_frame = frames['right_ir']
            elif not is_right and 'left_ir' in frames and frames['left_ir'] is not None:
                ir_frame = frames['left_ir']
            elif 'ir' in frames and frames['ir'] is not None:
                ir_frame = frames['ir']
            else:
                raise RuntimeError("No IR frame found in capture result")
                
            return ir_frame
            
        except Exception as e:
            logger.error(f"Error capturing {side} IR: {e}")
            raise

    async def capture_stereo_ir(self, 
                                exposure_val: int = 2000, 
                                gain_val: int = 16, 
                                light_mode: Literal['flood', 'laser', 'none'] = 'flood'):
        """捕获立体红外图像"""
        logger.info("Capturing stereo IR")
        try:
            # 先捕获左眼，再捕获右眼
            left_ir = await self.capture_ir(is_right=False, exposure_val=exposure_val, gain_val=gain_val, light_mode=light_mode)
            # 短暂延迟确保相机切换稳定
            await asyncio.sleep(0.1)
            right_ir = await self.capture_ir(is_right=True, exposure_val=exposure_val, gain_val=gain_val, light_mode=light_mode)
            
            logger.info(f"Successfully captured stereo IR images: left={left_ir.shape}, right={right_ir.shape}")
            return {'left_ir': left_ir, 'right_ir': right_ir}
        except Exception as e:
            logger.error(f"Error capturing stereo IR: {e}")
            raise

    async def capture_color(self, 
                        exposure_val: int = 100, 
                        gain_val: int = 1000,
                        auto_exposure: bool = False,
                        timeout_ms: int = 3000):
        """捕获彩色图像"""
        logger.info("Capturing color image")
        try:
            set_camera_params(self.dev, {'OB_PROP_COLOR_EXPOSURE_INT': exposure_val})
            set_camera_params(self.dev, {'OB_PROP_COLOR_GAIN_INT': gain_val})
            set_camera_params(self.dev, {'OB_PROP_COLOR_AUTO_EXPOSURE_BOOL': auto_exposure})
            
            # 确保启用彩色流
            self.config.enable_video_stream(OBSensorType.COLOR_SENSOR)
            
            frames = await self.capture_frames(timeout_ms)
            
            if 'color' not in frames or frames['color'] is None:
                raise RuntimeError("No color frame found in capture result")
            
            logger.info(f"Successfully captured color image: {frames['color'].shape}")
            return frames['color']
        
        except Exception as e:
            logger.error(f"Error capturing color: {e}")
            raise

class SyncOrbbecCamera:
    def __init__(self, camera_serial: str, pipeline_params: dict = {}, fps: int = 30):
        """
        初始化同步Orbbec相机
        
        Args:
            camera_serial: 相机序列号
            pipeline_params: 管道参数
            fps: 帧率
        """
        self.context = pyob.Context()
        self.camera_serial = camera_serial
        self.device = None
        self.pipeline = None
        self.dev = None
        self.config = pyob.Config()
        self.interval = 1000 // fps
        
        # 先查找设备
        self.find_device()
        
        # 初始化pipeline
        self.pipeline = pyob.Pipeline(self.device)
        self.dev = self.pipeline.get_device()
        
        # 设置管道配置
        set_pipeline_config(pipeline_params, config=self.config)

    def find_device(self):
        """根据序列号查找设备"""
        device_list = self.context.query_devices()
        if len(device_list) == 0:
            raise RuntimeError("No Orbbec devices found")

        # 根据序列号查找设备
        selected_device = None
        for device in device_list:
            device_info = device.get_device_info()
            if device_info.get_serial_number() == self.camera_serial:
                selected_device = device
                break
        
        if selected_device is None:
            available_serials = [device.get_device_info().get_serial_number() for device in device_list]
            raise ValueError(f"Camera with serial {self.camera_serial} not found. Available cameras: {available_serials}")
        
        self.device = selected_device
        device_info = self.device.get_device_info()
        logger.info(f"Sync Camera initialized: Serial: {device_info.get_serial_number()}, Name: {device_info.get_name()}")
        return self.device

    def flush_cache(self):
        """Clear camera cache by draining the frame buffer."""
        while self.pipeline.wait_for_frames(0) is not None:
            pass
        logger.debug("Camera cache flushed")

    def start(self):
        """启动相机流"""
        self.pipeline.start(self.config)
        time.sleep(0.1)

    def stop(self):
        self.pipeline.stop()

    def capture_frames(self, timeout_ms: int = 2000):
        self.start()
        self.flush_cache()
        frames = None
        start_t = time.time()
        while True:
            frames = self.pipeline.wait_for_frames(0)  # check is working
            if frames is not None:
                break
            time.sleep(self.interval / 1000)
            if time.time() - start_t > timeout_ms / 1000:
                break
        self.stop()

        if frames is None:
            raise RuntimeError(f"Failed to capture frame within {timeout_ms}ms")

        return convert_frameset(frames)

    def capture_ir(self, 
                   is_right: bool = False, 
                   exposure_val: int = 2000, 
                   gain_val: int = 16, 
                   light_mode: Literal['flood', 'laser', 'none'] = 'flood'):
        if is_right:
            chn = CHANNEL_IR_RIGHT
        else:
            chn = CHANNEL_IR_LEFT

        """捕获左侧红外图像"""
        try:
            set_camera_params(self.dev, {'OB_PROP_IR_CHANNEL_DATA_SOURCE_INT': chn})
            set_camera_params(self.dev, {'OB_PROP_IR_EXPOSURE_INT': exposure_val})
            set_camera_params(self.dev, {'OB_PROP_IR_GAIN_INT': gain_val})
            if light_mode == 'flood':
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': True})
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': False})
            elif light_mode == 'laser':
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': False})
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': True})
            elif light_mode == 'none':
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': False})
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': True})
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': False})
            else:
                logger.warning(f"Unknown light mode: {light_mode}, using default (flood)")
                set_camera_params(self.dev, {'OB_PROP_FLOOD_BOOL': True})
                set_camera_params(self.dev, {'OB_PROP_LASER_BOOL': False})

            frames = self.capture_frames()
            if 'ir' not in frames:
                raise RuntimeError("Failed to capture left IR frame")
            return frames['ir']
        except Exception as e:
            logger.error(f"Error capturing left IR: {e}")
            raise

    def capture_stereo_ir(self, 
                          exposure_val: int = 2000, 
                          gain_val: int = 16, 
                          light_mode: Literal['flood', 'laser', 'none'] = 'flood'):
        """捕获立体红外图像"""
        try:
            left_ir = self.capture_ir(is_right=False, exposure_val=exposure_val, gain_val=gain_val, light_mode=light_mode)
            right_ir = self.capture_ir(is_right=True, exposure_val=exposure_val, gain_val=gain_val, light_mode=light_mode)
            return {'left_ir': left_ir, 'right_ir': right_ir}
        except Exception as e:
            logger.error(f"Error capturing stereo IR: {e}")
            raise
    def capture_color(self, 
                    exposure_val: int = 100, 
                    gain_val: int = 1000,
                    auto_exposure: bool = False,
                    timeout_ms: int = 2000):
        """捕获彩色图像"""
        try:
            # 设置彩色相机参数
            set_camera_params(self.dev, {'OB_PROP_COLOR_EXPOSURE_INT': exposure_val})
            set_camera_params(self.dev, {'OB_PROP_COLOR_GAIN_INT': gain_val})
            set_camera_params(self.dev, {'OB_PROP_COLOR_AUTO_EXPOSURE_BOOL': auto_exposure})
            
            # 确保管道配置启用了彩色流
            self.config.enable_video_stream(OBSensorType.COLOR_SENSOR)
            
            # 捕获帧（capture_frames内部会刷缓存）
            frames = self.capture_frames(timeout_ms)
            
            if 'color' not in frames or frames['color'] is None:
                raise RuntimeError("Failed to capture color frame")
            
            return frames['color']
        
        except Exception as e:
            logger.error(f"Error capturing color: {e}")
            raise