import cv2
import numpy as np

class OptFlow:
    def __init__(self, resize_width=320, resize_height=180, height_start=0.2, height_end=0.5):
        self.width = resize_width
        self.height = resize_height

        self.height_start = int(self.height * height_start)
        self.height_end = int(self.height * height_end)

    def get_direction(self, frame1, frame2):
        frame1 = cv2.resize(frame1, (self.width, self.height))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2, (self.width, self.height))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1[self.height_start:self.height_end],
                                            frame2[self.height_start:self.height_end], None, 0.5, 3, 15, 1, 5, 1.2, 0)
        flow_avg = np.median(flow, axis=(0, 1))

        move_x = -1 * flow_avg[0]
        move_y = -1 * flow_avg[1]

        return move_x, move_y


if __name__ == '__main__':
    flow = OptFlow()

    cap = cv2.VideoCapture("test_video/video.mp4")
    _, img1 = cap.read()
    _, img2 = cap.read()

    features = []

    while True:
        x, y = flow.get_direction(img1, img2)
        STEER = x*50

        print('STEER: {}'.format(STEER))

        cv2.imshow('result', img1)
        wheel = cv2.imread('steering_wheel_image.jpg')
        wheel = cv2.resize(wheel, dsize=(335, 335), interpolation=cv2.INTER_AREA)

        M = cv2.getRotationMatrix2D((335 / 2, 335 / 2), -int(STEER), 1)

        wheel = cv2.warpAffine(wheel, M, (335, 335))

        cv2.imshow('wheel', wheel)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img1 = img2
        _, img2 = cap.read()
        img2 = cv2.resize(img2, dsize=(640, 360), interpolation=cv2.INTER_AREA)