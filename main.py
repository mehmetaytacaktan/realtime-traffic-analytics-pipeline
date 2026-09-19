import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def run_traffic_pipeline(
    source_video_path: str,
    target_video_path: str,
    start_second: int = 0,
    end_second: int = 15,
) -> None:
    """Executes object detection, multi-object tracking, and line-crossing counting pipeline."""
    # Initialize YOLOv8 object detection model
    model = YOLO("yolov8n.pt")

    # Initialize ByteTrack tracker and define COCO vehicle class IDs
    tracker = sv.ByteTrack()
    CLASS_ID_VEHICLES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

    # Initialize visual annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(trace_length=30)

    # Extract video metadata and calculate target frame bounds
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)

    # Configure line-crossing counting zone at 65% frame height
    line_start = sv.Point(50, int(video_info.height * 0.65))
    line_end = sv.Point(video_info.width - 50, int(video_info.height * 0.65))

    line_zone = sv.LineZone(start=line_start, end=line_end)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.8)

    print(
        f"Video Metadata: {video_info.width}x{video_info.height} @ {fps:.2f} FPS"
    )
    print(
        f"Processing Interval: {start_second}s -> {end_second}s (Frames: {start_frame} to {end_frame})"
    )

    frame_idx = 0

    # Stream processing loop
    with sv.VideoSink(
        target_path=target_video_path, video_info=video_info
    ) as sink:
        for frame in sv.get_video_frames_generator(
            source_path=source_video_path
        ):
            frame_idx += 1

            if frame_idx < start_frame:
                continue

            if frame_idx > end_frame:
                print(
                    f"\n[INFO] Target end time ({end_second}s) reached. Terminating process."
                )
                break

            # Object detection inference
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter for vehicle categories only
            detections = detections[
                np.isin(detections.class_id, CLASS_ID_VEHICLES)
            ]

            # Update ByteTrack state
            detections = tracker.update_with_detections(detections)

            # Evaluate line crossings
            line_zone.trigger(detections=detections)

            # Format visual labels
            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:.2f}"
                for confidence, class_id, tracker_id in zip(
                    detections.confidence,
                    detections.class_id,
                    detections.tracker_id,
                )
            ]

            # Render visual overlays
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

            # Real-time preview window
            cv2.imshow("Realtime Traffic Analytics", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Interrupted by user.")
                break

    cv2.destroyAllWindows()

    print("\n--- PROCESSING COMPLETE ---")
    print(f"Inbound Count:  {line_zone.in_count}")
    print(f"Outbound Count: {line_zone.out_count}")
    print(
        f"Total Vehicles: {line_zone.in_count + line_zone.out_count}"
    )
    print(f"Exported Video: {target_video_path}")


if __name__ == "__main__":
    run_traffic_pipeline(
        source_video_path="sample_traffic.mp4",
        target_video_path="output_analytics.mp4",
        start_second=5,
        end_second=20,
    )