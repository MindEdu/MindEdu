# %%
import time
import cv2

# %%
from ultralytics import YOLO

# %%
def process_video(input_video_path, output_video_path, model_path, frame_resize=None):
    # 加载模型
    model = YOLO(model_path)

    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_resize:
        frame_width, frame_height = frame_resize

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 逐帧处理
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取结束

        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total_frames}")

        # 可选：调整帧大小
        if frame_resize:
            frame = cv2.resize(frame, frame_resize)

        # 使用 YOLO 模型进行目标检测
        results = model(frame)

        # 绘制检测结果
        annotated_frame = results[0].plot()

        # 写入到输出视频
        out.write(annotated_frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"Video processing completed. Output saved to {output_video_path}")


start_time = time.time()
# %%
process_video('example.mp4', 'output.mp4', 'best.pt', frame_resize=None)

end_time = time.time()

print(end_time-start_time)
