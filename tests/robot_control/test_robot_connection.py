from robot_control.robot_connection import RobotClient, AsyncRobotMessageConnection, AsyncRobotConnection
import time
import struct
from pose_transformation import transform_1x6_to_4x4, transform_4x4_to_1x6, get_upper_pose
from robot_control.robot_control import calc_pose_diff, AsyncRobotClient

async def test():
    # robot = AsyncRobotMessageConnection("192.168.0.105", 30001, '../../data/robot_arm/RobotStateMessage.xlsx:v2.6.0')
    # robot_req = AsyncRobotConnection("192.168.0.105", 40011)
    # await robot_req.connect()
    # await robot_req.send(b"req 1 get_actual_tcp_speed()\n")
    # response = await robot_req.recv()
    # if response is not None:
    #     # size, type = struct.unpack_from('>iB', response)
    #     # print("size, type:", size, type)
    #     print("Response:", response.decode('utf-8'))

    # await robot.connect()
    # pose = None
    # while pose is None:
    #     data = await robot.recv_data()
    #     # print("data:", data)
    #     if data is None:
    #         print("No data received")
    #         continue
    #     pose = [
    #         data['tcp_x'] * 1000,
    #         data['tcp_y'] * 1000,
    #         data['tcp_z'] * 1000,
    #         data['rot_x'],
    #         data['rot_y'],
    #         data['rot_z']
    #     ]
    #     print("TCP Pose:", pose)
    robot_client = AsyncRobotClient("192.168.0.105", 40011, 30001, {"a": 1.0,
      "r": 0.0,
      "t": 0.0,
      "v": 0.2}, {
      "a": 1.0,
      "r": 0.0,
      "t": 0.0,
      "v": 0.2
    }, 0.01, 0.01)
    await robot_client.connect()
    pose = await robot_client.get_target_tcp_pose()
    print("Current Pose:", pose)
    pose_1x6 = transform_4x4_to_1x6(pose)
    print("Current Pose 1x6:", pose_1x6)
    # pose_1x6 = [pose_1x6[0] *1000, pose_1x6[1] *1000, pose_1x6[2] *1000, pose_1x6[3], pose_1x6[4], pose_1x6[5]]
    print(transform_1x6_to_4x4([0,0,0,0,0,0]))
    # print(transform_4x4_to_1x6(get_upper_new_pose(pose, offset=100)))
    # speed = await robot_client.get_target_tcp_speed()
    # print("Current Speed:", speed)
    # await robot_client.move("movel", [
    #     -119.756,
    #     602.91,
    #     624.7270000000001,
    #     0.288796,
    #     -0.076242,
    #     -3.131485
    # ], None, True)

async def main():
    await test()

if __name__ == "__main__":
    import anyio
    anyio.run(main)
    # robot_client = RobotClient("192.168.0.105", 30001, '../../data/robot_arm/RobotStateMessage.xlsx:v2.6.0')
    # pose = None
    # while pose is None:
    #     data = robot_client.recv_data()
    #     if data is None:
    #         print("No data received")
    #         continue
    #     pose = [
    #         data['tcp_x'] * 1000,
    #         data['tcp_y'] * 1000,
    #         data['tcp_z'] * 1000,
    #         data['rot_x'],
    #         data['rot_y'],
    #         data['rot_z']
    #     ]
    #     print("TCP Pose:", pose)
        # time.sleep(2)
    # mat1 = [-119.75565627106802, 602.9108541324916, 624.7242591906866, 0.2888006506934992, -0.07624153496589613, -3.131485592407821]
    # mat2 = [-120.75565627106802, 605.9108541324916, 679.7242591906866, 0.3888006506934992, -0.07624153496589613, -3.131485592407821]
    # print(calc_pose_diff(mat1, mat2))
    

    # res = transform_1x6_to_4x4(mat)
    # print(res)
    # print(transform_4x4_to_1x6(res))