import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.85, track_con=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, max_hands, 1, detection_con, track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def fingers_up(self):
        fingers = []
         
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers (Check if tip is above joint)
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers