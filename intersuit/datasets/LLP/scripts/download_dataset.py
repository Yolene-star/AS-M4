import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Download LLP videos with an optional video-id whitelist.")
    parser.add_argument("--source", default="data/AVVP_dataset_full.csv")
    parser.add_argument("--output-root", default="data/LLP_dataset")
    parser.add_argument("--whitelist", default=None, help="Text file with one YouTube video id per line.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of unique whitelisted videos to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected videos without downloading.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep processing remaining videos when one download fails.")
    parser.add_argument("--report", default=None, help="Optional JSONL report path for per-video download status.")
    return parser.parse_args()


def load_whitelist(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    return set(values)


def resolve_downloader():
    for name in ("youtube-dl", "yt-dlp"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("Neither youtube-dl nor yt-dlp was found in PATH.")


def run_checked(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        details = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise RuntimeError(details[-4000:] if details else "command failed with exit code %s" % result.returncode)

def download(set, name, t_seg, downloader):
    #label = label.replace(" ", "_")  # avoid space in folder name
    path_data = os.path.join(set, "video")
    print(path_data)
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    link_prefix = "https://www.youtube.com/watch?v="

    filename_full_video = os.path.join(path_data, name) + "_full_video.mp4"
    filename = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print("already exists, skip")
        return

    print( "download the whole video for: [%s] - [%s]" % (set, name))
    command1 = [
        downloader,
        "--ignore-config",
        link,
        "-o",
        filename_full_video,
        "-f",
        "b",
    ]
    run_checked(command1)

    t_start, t_end = t_seg
    t_dur = t_end - t_start
    print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
    command2 = [
        "ffmpeg",
        "-ss",
        str(t_start),
        "-i",
        filename_full_video,
        "-t",
        str(t_dur),
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-strict",
        "-2",
        filename,
        "-y",
        "-loglevel",
        "-8",
    ]
    run_checked(command2)
    try:
        os.remove(filename_full_video)
    except:
        return

    print ("finish the video as: " + filename)
    return filename


def write_report_row(path, row):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


##%% read the label encoding
# filename = "../doc/class_labels_indices.csv"
# lines = [x.strip() for x in open(filename, 'r')][1:]
# label_encode = {}
# for l in lines:
#    l = l[l.find(",")+1:]
#    encode = l.split(",")[0]
#    label_encode[ l[len(encode)+2:-1] ] = encode
#
#
#

def main():
    args = parse_args()
    if args.report and os.path.exists(args.report):
        os.remove(args.report)
    whitelist = load_whitelist(args.whitelist)
    df = pd.read_csv(args.source, header=0, sep="\t")
    filenames = df["filename"]
    print(len(filenames))
    names = []
    segments = {}
    selected = 0
    for i in range(len(filenames)):
        row = df.loc[i, :]
        filename = row.iloc[0]
        name = filename[:11]
        if whitelist is not None and name not in whitelist:
            continue
        if name in segments:
            continue
        if args.limit is not None and selected >= args.limit:
            break
        steps = filename[11:].split("_")
        t_start = float(steps[1])
        t_end = t_start + 10
        segments[name] = (t_start, t_end)
        names.append(name)
        selected += 1
        if args.dry_run:
            print("selected %s %.1f %.1f" % (name, t_start, t_end))
        else:
            try:
                output_path = download(args.output_root, name, segments[name], resolve_downloader())
                write_report_row(
                    args.report,
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "video_id": name,
                        "segment_start": t_start,
                        "segment_end": t_end,
                        "ok": True,
                        "output_path": output_path,
                        "error": None,
                    },
                )
            except Exception as exc:
                write_report_row(
                    args.report,
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "video_id": name,
                        "segment_start": t_start,
                        "segment_end": t_end,
                        "ok": False,
                        "output_path": None,
                        "error": str(exc),
                    },
                )
                if not args.continue_on_error:
                    raise
    print(len(segments))


if __name__ == "__main__":
    main()
