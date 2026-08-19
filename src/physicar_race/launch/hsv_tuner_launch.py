"""차선 HSV 대화형 튜너.

주행 스택과 따로 띄운다. 카메라만 있으면 되고 차는 안 움직인다.

    ros2 launch physicar_race hsv_tuner_launch.py
    ros2 launch physicar_race hsv_tuner_launch.py image_topic:=/image_raw

슬라이더로 맞춘 뒤 [s] 를 누르면 그대로 쓸 수 있는 launch 인자가 출력된다.

디스플레이가 없으면 창 대신 tuner/image 토픽으로 발행하고, 값은
ros2 param set /hsv_tuner_node <이름> <값> 으로 바꾼다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        # 4분할이라 창이 커진다. 화면이 작으면 더 줄일 것.
        DeclareLaunchArgument('scale', default_value='0.5'),
        # 디스플레이가 있어도 강제로 토픽 발행만 하고 싶을 때
        DeclareLaunchArgument('force_headless', default_value='false'),
    ]

    tuner = Node(
        package='physicar_race',
        executable='hsv_tuner_node',
        name='hsv_tuner_node',
        output='screen',
        parameters=[{
            'scale': ParameterValue(LaunchConfiguration('scale'), value_type=float),
            'force_headless': ParameterValue(
                LaunchConfiguration('force_headless'), value_type=bool),
        }],
        remappings=[('image_raw', LaunchConfiguration('image_topic'))],
    )

    return LaunchDescription(args + [tuner])
