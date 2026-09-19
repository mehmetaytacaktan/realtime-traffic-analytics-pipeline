import os
import tempfile
import gradio as gr
import imageio_ffmpeg as ffmpeg
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load YOLOv8 model and filter for vehicles (car, motorcycle, bus, truck)
MODEL = YOLO("yolov8n.pt")
CLASS_ID_VEHICLES = [2, 3, 5, 7]

SAMPLE_VIDEO_PATH = "sample_traffic.mp4"


def process_video(input_video_path):
    if input_video_path is None:
        return None

    raw_output_path = tempfile.NamedTemporaryFile(
        delete=False, suffix="_raw.mp4"
    ).name
    web_output_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    ).name

    video_info = sv.VideoInfo.from_video_path(input_video_path)

    # Initialize tracking and annotation modules
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(trace_length=30)

    # Counting line setup
    line_start = sv.Point(50, int(video_info.height * 0.65))
    line_end = sv.Point(video_info.width - 50, int(video_info.height * 0.65))
    line_zone = sv.LineZone(start=line_start, end=line_end)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.8)

    with sv.VideoSink(
        target_path=raw_output_path, video_info=video_info
    ) as sink:
        for frame in sv.get_video_frames_generator(
            source_path=input_video_path
        ):
            results = MODEL(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[
                np.isin(detections.class_id, CLASS_ID_VEHICLES)
            ]
            detections = tracker.update_with_detections(detections)
            line_zone.trigger(detections=detections)

            labels = [
                f"#{tracker_id} {MODEL.model.names[class_id]}"
                for class_id, tracker_id in zip(
                    detections.class_id, detections.tracker_id
                )
            ]

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            annotated_frame = line_zone_annotator.annotate(
                frame=annotated_frame, line_counter=line_zone
            )

            sink.write_frame(annotated_frame)

    # Transcode video to H.264 format for HTML5 browser compatibility
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    os.system(
        f'{ffmpeg_exe} -y -i "{raw_output_path}" -vcodec libx264 "{web_output_path}" -loglevel quiet'
    )

    if os.path.exists(raw_output_path):
        os.remove(raw_output_path)

    return web_output_path


# Gradio UI Construction
with gr.Blocks(title="Real-Time Traffic Analytics Pipeline") as demo:
    gr.Markdown("# Real-Time Traffic Analytics Pipeline")
    gr.Markdown(
        "Object detection, multi-object tracking (MOT), and vehicle counting demo powered by YOLOv8 and ByteTrack."
    )

    with gr.Row():
        input_video = gr.Video(
            label="Input Video (Upload your file or select a sample below)",
            value=SAMPLE_VIDEO_PATH
            if os.path.exists(SAMPLE_VIDEO_PATH)
            else None,
        )
        output_video = gr.Video(
            label="Processed Output Stream", autoplay=True
        )

    submit_btn = gr.Button("Run Pipeline", variant="primary")

    if os.path.exists(SAMPLE_VIDEO_PATH):
        gr.Examples(
            examples=[[SAMPLE_VIDEO_PATH]],
            inputs=input_video,
            outputs=output_video,
            fn=process_video,
            cache_examples=False,
            label="Sample Demo Videos",
        )

    submit_btn.click(
        fn=process_video, inputs=input_video, outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)