import cv2
from deepface import DeepFace

class VideoEmotionAnalyzer:
    def __init__(self):
        # DeepFace loads models automatically on first run
        pass

    def analyze_frame(self, frame):
        try:
            # DeepFace expects BGR (OpenCV standard) or RGB
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            return result[0]['dominant_emotion']
        except:
            return "neutral"

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        emotions = []
        frame_count = 0
        
        # Analyze every 30th frame (1 per second for 30fps video) to save time
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:
                emotion = self.analyze_frame(frame)
                emotions.append(emotion)
            
            frame_count += 1
            
        cap.release()
        
        if not emotions:
            return "neutral"
            
        # Find most frequent emotion
        return max(set(emotions), key=emotions.count)