import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detConf=0.5, trConf=0.5):
        self.mode = mode
        self.maxHands = maxHands  # Fixed typo here
        self.detConf = detConf
        self.trConf = trConf
        self.mpHands = mp.solutions.hands.Hands(self.mode, self.maxHands, self.detConf, self.trConf)  # Fixed typo here
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mpHands.process(frame_rgb)  # Fixed 'hands' to 'self.mpHands'
        if draw:
            self.mpDraw.draw_landmarks(frame, results.multi_hand_landmarks, self.mpHands.HAND_CONNECTIONS)  # Fixed 'landmarks' to 'results.multi_hand_landmarks'
        return frame

def main():
    camera_capture = cv2.VideoCapture(0)

    while camera_capture.isOpened():  # Added .isOpened() to the condition
        ret, frame = camera_capture.read()
        if not ret:
            print("Failed to retrieve frame.")
            break
    
        detector = HandDetector()
        img = detector.FindHands(camera_capture)
        cv2.namedWindow("hand tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("hand tracking", img)
        if cv2.waitKey(1):
            break

    

if __name__ == "__main__":  # Fixed '=' to '=='
    main()
