import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build voiceassistant.json and audio cache from VoiceAssistant parquet files.")
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--limit", type=int, default=5000, help="Max number of samples to export. Use <=0 for all.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    parquet_dir = project_root / "VoiceAssistant-400K" / "data"
    audio_dir = project_root / "VoiceAssistant-400K" / "audios"
    output_json = project_root / "voiceassistant.json"

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Missing parquet directory: {parquet_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")

    records = []
    seen = set()
    written_audio = 0
    scanned_rows = 0

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, columns=["question", "question_audio"])
        for _, row in df.iterrows():
            scanned_rows += 1
            question = row.get("question")
            qa = row.get("question_audio")
            if not isinstance(question, str) or not isinstance(qa, dict):
                continue

            speech_name = qa.get("path")
            speech_bytes = qa.get("bytes")
            if not speech_name or not isinstance(speech_name, str):
                continue
            if speech_name in seen:
                continue

            target_wav = audio_dir / speech_name
            if isinstance(speech_bytes, (bytes, bytearray)) and not target_wav.exists():
                target_wav.write_bytes(speech_bytes)
                written_audio += 1

            records.append({"speech": speech_name, "question": question})
            seen.add(speech_name)

            if args.limit > 0 and len(records) >= args.limit:
                break
        if args.limit > 0 and len(records) >= args.limit:
            break

    if not records:
        raise RuntimeError("No usable records were extracted from parquet files.")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Parquet files scanned: {len(parquet_files)}")
    print(f"Rows scanned: {scanned_rows}")
    print(f"Records exported: {len(records)}")
    print(f"Audio files written: {written_audio}")
    print(f"Output JSON: {output_json}")
    print(f"Audio directory: {audio_dir}")


if __name__ == "__main__":
    main()
