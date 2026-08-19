#!/usr/bin/env python3
"""차선 HSV 대화형 튜너 - 슬라이더로 바로 맞춘다.

probe 로그로 값을 읽고 다시 launch 하는 왕복이 느려서, 화면을 보면서
슬라이더로 즉시 맞출 수 있게 만들었다.

화면은 2x2 로 나온다:
    원본 + ROI 경계    |  흰선 마스크
    노란선 마스크      |  오버레이 (흰=흰선, 노랑=노란선)

키:
    s  현재 값을 launch 인자 한 줄로 출력 (복사해서 쓰면 된다)
    r  기본값으로 되돌림
    q  종료

디스플레이가 없는 환경(headless 컨테이너 등)이면 창을 못 띄우므로,
자동으로 'tuner/image' 토픽 발행으로 전환한다. 그 경우 값 조정은
ros2 param set 으로 한다 -- 이 노드는 파라미터를 실시간으로 반영한다.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

WINDOW = 'HSV tuner  [s]=값출력  [r]=초기화  [q]=종료'

# (파라미터 이름, 슬라이더 최대값, 기본값)
CONTROLS = [
    ('roi_top_pct', 100, 55),      # ROI 시작 위치 (%)
    ('white_s_max', 255, 60),
    ('white_v_min', 255, 180),
    ('yellow_h_min', 179, 18),
    ('yellow_h_max', 179, 38),
    ('yellow_s_min', 255, 90),
    ('yellow_v_min', 255, 90),
]


def build_masks(roi_bgr, v):
    """ROI 에서 흰선/노란선 마스크를 만든다. GUI 없이도 테스트 가능하도록 순수 함수."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, v['white_v_min']), (180, v['white_s_max'], 255))
    yellow = cv2.inRange(
        hsv,
        (v['yellow_h_min'], v['yellow_s_min'], v['yellow_v_min']),
        (v['yellow_h_max'], 255, 255),
    )
    k = np.ones((3, 3), np.uint8)
    return (cv2.morphologyEx(white, cv2.MORPH_OPEN, k),
            cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k))


def render(bgr, v):
    """2x2 합성 화면을 만든다. 검출된 선 개수도 같이 돌려준다."""
    h, w = bgr.shape[:2]
    y0 = int(h * v['roi_top_pct'] / 100.0)
    y0 = max(0, min(h - 2, y0))
    roi = bgr[y0:, :]

    white, yellow = build_masks(roi, v)

    orig = bgr.copy()
    cv2.line(orig, (0, y0), (w, y0), (0, 0, 255), 2)
    cv2.putText(orig, 'ROI', (8, max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 마스크는 ROI 크기라 원본 크기 캔버스에 얹어 4분할 크기를 맞춘다
    def to_canvas(mask, color):
        c = np.zeros((h, w, 3), np.uint8)
        c[y0:, :][mask > 0] = color
        return c

    white_v = to_canvas(white, (255, 255, 255))
    yellow_v = to_canvas(yellow, (0, 255, 255))

    overlay = bgr.copy()
    overlay[y0:, :][white > 0] = (255, 255, 255)
    overlay[y0:, :][yellow > 0] = (0, 255, 255)

    n_white = cv2.connectedComponentsWithStats(white, connectivity=8)[0] - 1
    n_yellow = cv2.connectedComponentsWithStats(yellow, connectivity=8)[0] - 1
    cv2.putText(overlay, 'white=%d  yellow=%d' % (n_white, n_yellow),
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    top = np.hstack([orig, white_v])
    bottom = np.hstack([yellow_v, overlay])
    return np.vstack([top, bottom]), n_white, n_yellow


def normalize(values):
    """yellow_h_min > max 면 inRange 가 빈 마스크를 낸다.

    슬라이더를 돌리다 보면 쉽게 넘어가는데, 결과가 '아무것도 안 잡힘'이라
    임계값 문제인지 구간 역전인지 구분이 안 된다. 여기서 바로잡고 알린다.
    """
    v = dict(values)
    swapped = v['yellow_h_min'] > v['yellow_h_max']
    if swapped:
        v['yellow_h_min'], v['yellow_h_max'] = v['yellow_h_max'], v['yellow_h_min']
    return v, swapped


def launch_args(v):
    """지금 값을 그대로 붙여 넣을 수 있는 launch 인자 한 줄."""
    return (
        'ros2 launch physicar_race race_launch.py \\\n'
        '  lane_roi_top_frac:=%.2f \\\n'
        '  lane_white_s_max:=%d lane_white_v_min:=%d \\\n'
        '  lane_yellow_h_min:=%d lane_yellow_h_max:=%d \\\n'
        '  lane_yellow_s_min:=%d lane_yellow_v_min:=%d'
        % (v['roi_top_pct'] / 100.0,
           v['white_s_max'], v['white_v_min'],
           v['yellow_h_min'], v['yellow_h_max'],
           v['yellow_s_min'], v['yellow_v_min'])
    )


class HsvTunerNode(Node):
    def __init__(self):
        super().__init__('hsv_tuner_node')

        for name, _max, default in CONTROLS:
            self.declare_parameter(name, default)
        self.declare_parameter('scale', 0.5)      # 4분할이라 창이 커진다
        self.declare_parameter('force_headless', False)

        self.scale = float(self.get_parameter('scale').value)
        self.values = {n: int(self.get_parameter(n).value) for n, _m, _d in CONTROLS}

        self.bridge = CvBridge()
        self.gui = not bool(self.get_parameter('force_headless').value)
        self._warned = False

        if self.gui:
            self.gui = self._try_open_window()

        self.pub = self.create_publisher(Image, 'tuner/image', 1)
        self.create_subscription(Image, 'image_raw', self.on_image, qos_profile_sensor_data)

        if self.gui:
            self.get_logger().info('슬라이더 창을 띄웠다. [s]=값출력 [r]=초기화 [q]=종료')
        else:
            self.get_logger().warn(
                '디스플레이가 없어 창을 못 띄운다. tuner/image 토픽으로 발행한다.\n'
                '  값 조정: ros2 param set /hsv_tuner_node white_v_min 120\n'
                '  화면 보기: ros2 run rqt_image_view rqt_image_view')

    def _try_open_window(self):
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            for name, maxv, _d in CONTROLS:
                cv2.createTrackbar(name, WINDOW, self.values[name], maxv,
                                   lambda x, n=name: self._on_track(n, x))
            return True
        except Exception as e:
            self.get_logger().warn('창 생성 실패(%s) - headless 모드로 전환' % e)
            return False

    def _on_track(self, name, value):
        self.values[name] = int(value)

    def _sync_from_params(self):
        """headless 모드에서 ros2 param set 을 실시간 반영한다."""
        for name, _m, _d in CONTROLS:
            self.values[name] = int(self.get_parameter(name).value)

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('이미지 변환 실패: %s' % e, throttle_duration_sec=2.0)
            return

        if not self.gui:
            self._sync_from_params()

        v, swapped = normalize(self.values)
        if swapped and not self._warned:
            self.get_logger().warn('yellow_h_min > max 라 값을 뒤집어 적용했다')
            self._warned = True

        canvas, n_w, n_y = render(bgr, v)

        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)

        out = self.bridge.cv2_to_imgmsg(canvas, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)

        if not self.gui:
            return

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.get_logger().info('\n' + launch_args(v))
        elif key == ord('r'):
            for name, _m, default in CONTROLS:
                self.values[name] = default
                cv2.setTrackbarPos(name, WINDOW, default)
            self.get_logger().info('기본값으로 되돌렸다')
        elif key == ord('q'):
            self.get_logger().info('\n' + launch_args(v))
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = HsvTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
