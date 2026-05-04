import cv2 as cv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FRAMES_TO_SAVE = [1, 70, 210, 350]

VIDEO_OPENCV = "processed/frisbee_opencv.mp4"
VIDEO_MINE   = "processed/frisbee_custom.mp4"

OUTPUT_DIR   = "frames"
OUTPUT_PLOT  = "frames/comparison.png"


def open_video(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити відео: {path}")
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return cap, total


def read_frame(cap, fn):
    cap.set(cv.CAP_PROP_POS_FRAMES, fn - 1)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


def save_frame(frame_rgb, path):
    cv.imwrite(path, cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))


def extract_and_compare(video_opencv, video_mine, frame_numbers, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("Відкриваємо відео...")
    cap_cv, total_cv = open_video(video_opencv)
    cap_mn, total_mn = open_video(video_mine)

    valid_frames = sorted(
        fn for fn in set(frame_numbers)
        if 1 <= fn <= min(total_cv, total_mn)
    )

    skipped = set(frame_numbers) - set(valid_frames)
    for fn in sorted(skipped):
        print(f"[!] Кадр {fn} поза межами, пропускаємо")

    if not valid_frames:
        print("Немає валідних кадрів. Завершення.")
        cap_cv.release()
        cap_mn.release()
        return


    frames_cv = {}
    frames_mn = {}

    for fn in valid_frames:
        f_cv = read_frame(cap_cv, fn)
        f_mn = read_frame(cap_mn, fn)

        if f_cv is None or f_mn is None:
            print(f"[!] Не вдалося прочитати кадр {fn}, пропускаємо")
            continue

        frames_cv[fn] = f_cv
        frames_mn[fn] = f_mn

        # зберігаємо окремі файли
        save_frame(f_cv, os.path.join(out_dir, f"frame_{fn:05d}_opencv.png"))
        save_frame(f_mn, os.path.join(out_dir, f"frame_{fn:05d}_mine.png"))
        print(f"  Збережено кадр {fn}")

    cap_cv.release()
    cap_mn.release()

    n = len(frames_cv)
    if n == 0:
        print("Нічого відображати.")
        return

    fig = plt.figure(figsize=(6, 2 * n))
    gs  = gridspec.GridSpec(
        nrows=n, ncols=2,
        hspace=0.35, wspace=0.05,
        left=0.03, right=0.97,
        top=0.97, bottom=0.03,
    )

    for row, fn in enumerate(sorted(frames_cv.keys())):
        # ліва колонка — OpenCV
        ax_cv = fig.add_subplot(gs[row, 0])
        ax_cv.imshow(frames_cv[fn])
        ax_cv.axis("off")
        ax_cv.set_title(
            f"Кадр №{fn}\nOpenCV реалізація",
            fontsize=11,  pad=6,
        )

        # права колонка — Custom
        ax_mn = fig.add_subplot(gs[row, 1])
        ax_mn.imshow(frames_mn[fn])
        ax_mn.axis("off")
        ax_mn.set_title(
            f"Кадр №{fn}\nВласна реалізація",
            fontsize=11, pad=6,
        )

    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    extract_and_compare(VIDEO_OPENCV, VIDEO_MINE, FRAMES_TO_SAVE, OUTPUT_DIR)