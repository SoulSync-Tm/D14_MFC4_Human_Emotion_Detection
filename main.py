import os
from moviepy.editor import VideoFileClip
from app.core.audio_processor import AudioEmotionAnalyzer
from app.core.video_processor import VideoEmotionAnalyzer

def process_media(file_path):
    print(f"Processing: {file_path}")
    
    # 1. Setup
    audio_analyzer = AudioEmotionAnalyzer()
    video_analyzer = VideoEmotionAnalyzer()
    
    # 2. Extract Audio from Video
    temp_audio = "temp_audio.wav"
    try:
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(temp_audio, logger=None)
    except Exception as e:
        print(f"Could not extract audio: {e}")
        return

    # 3. Analyze Audio
    print("Analyzing Audio...")
    audio_result = audio_analyzer.analyze_audio(temp_audio)
    print(f"Audio Result: {audio_result}")

    # 4. Analyze Video
    print("Analyzing Video Frames...")
    video_emotion = video_analyzer.analyze_video(file_path)
    print(f"Video Result: {video_emotion}")

    # 5. Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)

    # 6. Final Report
    print("\n--- FINAL EMOTION REPORT ---")
    print(f"Visual Emotion (Face): {video_emotion}")
    print(f"Vocal Emotion (Tone):  {audio_result['audio_emotion']} ({audio_result['audio_confidence']*100}%)")
    print("----------------------------")

if __name__ == "__main__":
    # Change this to your video file path
    # You can put a test video in the 'input' folder
    video_file = "input/test.mp4" 
    
    if os.path.exists(video_file):
        process_media(video_file)
    else:
        print(f"Please place a video file at {video_file} to test.")