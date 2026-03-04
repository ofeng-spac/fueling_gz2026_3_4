import json
import asyncio
from loguru import logger
from redis.asyncio import Redis
from typing import Optional

class FuelingController:
    def __init__(self, arm_id: int, robot_client, upper_pose, init_pose):
        self.arm_id = arm_id
        self.robot_client = robot_client
        self.upper_pose = upper_pose
        self.init_pose = init_pose
        self.fueling_complete_event = asyncio.Event()
        self.is_fueling = False
   


    async def wait_for_fueling_complete(self, redis_client, timeout=300):
        """等待加注完成信号"""
        channel = f"fueling_complete_arm_{self.arm_id}"
        logger.info(f"机械臂 {self.arm_id} 开始监听加注完成信号，频道: {channel}")
        
        # 使用pubsub监听频道
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            # 设置超时
            async with asyncio.timeout(timeout):
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        data = json.loads(message['data'])
                        if data.get('arm_id') == self.arm_id:
                            logger.info(f"机械臂 {self.arm_id} 收到加注完成信号: {data}")
                            return True
        except asyncio.TimeoutError:
            logger.warning(f"机械臂 {self.arm_id} 等待加注完成信号超时")
            raise
        finally:
            await pubsub.unsubscribe(channel)
    
    async def execute_retraction(self):
        """执行回退操作"""
        logger.info(f"机械臂 {self.arm_id} 开始执行回退操作")
        
        try:
            # 移动到upper_pose
            await self.robot_client.move(self.upper_pose)
            logger.info(f"机械臂 {self.arm_id} 已移动到upper_pose")
            
            # 移动到init_pose
            await self.robot_client.move(self.init_pose)
            logger.info(f"机械臂 {self.arm_id} 已移动到init_pose，回退完成")
            
        except Exception as e:
            logger.error(f"机械臂 {self.arm_id} 回退操作失败: {e}")
    
    async def start_fueling_monitor(self, redis_client: Redis):
        """启动加注监控"""
        self.is_fueling = True
        self.fueling_complete_event.clear()
        
        # 为每个机械臂分配不同的Redis频道
        channel = f"fueling_complete_arm_{self.arm_id}"
        
        # 启动监听任务
        listener_task = asyncio.create_task(
            self.wait_for_fueling_complete(redis_client, channel)
        )
        
        try:
            # 等待加注完成信号
            await self.fueling_complete_event.wait()
            logger.info(f"机械臂 {self.arm_id} 检测到加注完成，准备回退")
            
            # 启动回退任务（不等待完成，立即返回）
            await asyncio.create_task(self.execute_retraction())
            
        except Exception as e:
            logger.error(f"机械臂 {self.arm_id} 加注监控异常: {e}")
        finally:
            self.is_fueling = False
            if not listener_task.done():
                listener_task.cancel()

class RedisManager:
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 password: Optional[str] = None, db: int = 0):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.redis_client: Optional[Redis] = None
    
    async def connect(self):
        """连接Redis服务器"""
        try:
            self.redis_client = Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True
            )
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis连接成功")
            return True
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.redis_client = None
            return False
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis连接已关闭")
    
    async def publish_fueling_complete(self, arm_id: int):
        """发布加注完成信号"""
        if not self.redis_client:
            logger.warning("Redis客户端未连接，无法发布消息")
            return False
        
        try:
            channel = f"fueling_complete_arm_{arm_id}"
            message = json.dumps({"arm_id": arm_id, "status": "complete"})
            await self.redis_client.publish(channel, message)
            logger.info(f"已发布加注完成信号到频道 {channel}")
            return True
        except Exception as e:
            logger.error(f"发布加注完成信号失败: {e}")
            return False
    
    def create_fueling_controller(self, arm_id: int, robot_client, upper_pose, init_pose):
        """创建加注控制器"""
        return FuelingController(arm_id, robot_client, upper_pose, init_pose)
    
    @property
    def is_connected(self):
        """检查Redis是否连接"""
        return self.redis_client is not None