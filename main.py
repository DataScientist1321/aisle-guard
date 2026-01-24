import cv2
from detector.stream import VideoStream
from detector.sampler import FrameSampler


def main():
    MODE = "video"  # "video" or "live"

    if MODE == "video":
        source = "test_sample.mp4"
        loop_video = True
    else:
        source = 0
        loop_video = False

    stream = VideoStream(
        source=source,
        loop_video=loop_video
    )

    sampler = FrameSampler(process_every_n_frames=5)

    frame_count = 0

    for frame in stream.frames():
        frame_count += 1

        process_frame = sampler.should_process()

        if process_frame:
            cv2.putText(
                frame,
                "PROCESSING FRAME",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("Aisle Guard - Step 2 Test", frame)

        print(
            f"Frame: {frame_count} | Processing: {process_frame}",
            end="\r"
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
