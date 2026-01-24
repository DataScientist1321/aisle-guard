import cv2
from detector.stream import VideoStream
from detector.sampler import FrameSampler
from detector.tracker import PersonTracker
from detector.ring_buffer import RingBuffer


def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


def print_debug_tracks(tracks):
    print("\n--- TRACK DEBUG ---")
    for track in tracks:
        track_id = track["track_id"]
        center = track["center"]
        history = track["history"]

        print(f"Person ID: {track_id}")
        print(f"  Current center: {center}")
        print(f"  History length: {len(history)}")

        if len(history) > 1:
            recent = history[-5:]
            print("  Recent positions:")
            for h in recent:
                print(f"    Frame {h['frame_id']} -> Center {h['center']}")
        else:
            print("  Not enough history yet")


def main():
    MODE = "live"  # "video" or "live"

    if MODE == "video":
        source = "test_sample.mp4"
        loop_video = True
    else:
        source = 0
        loop_video = False

    stream = VideoStream(source=source, loop_video=loop_video)

    FPS = 30
    BUFFER_SECONDS = 15

    ring_buffer = RingBuffer(
        max_seconds=BUFFER_SECONDS,
        fps=FPS
    )

    sampler = FrameSampler(process_every_n_frames=5)
    tracker = PersonTracker(model_path="yolov8n.pt")

    frame_id = 0
    last_tracks = []

    for frame in stream.frames():
        frame_id += 1

        # Always add frame to ring buffer
        ring_buffer.add(frame)

        # Ring buffer sanity print
        if frame_id % 100 == 0:
            print(
                f"Ring buffer frames: {len(ring_buffer.get_frames())} / {ring_buffer.max_frames}"
            )

        # Run tracking only on sampled frames
        if sampler.should_process():
            last_tracks = tracker.track(frame, frame_id)
            print_debug_tracks(last_tracks)

        draw_tracks(frame, last_tracks)

        cv2.imshow("Aisle Guard - Step 5 Ring Buffer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
