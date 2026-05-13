import os
import re
import uuid
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from groq import Groq

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Groq Whisper 单次上传限制 25MB
MAX_FILE_SIZE = 25 * 1024 * 1024
# 分片时长（秒），保证每片 < 25MB
CHUNK_DURATION = 600  # 10 分钟


def download_audio(url: str, out_dir: str) -> str:
    """用 yt-dlp 下载最低码率音频，再用 ffmpeg 转为 mono 16kHz opus 进一步压缩."""
    raw_path = os.path.join(out_dir, "raw_audio")
    # 下载最低码率的纯音频
    cmd_dl = [
        "yt-dlp",
        "-f", "worstaudio",
        "-x",
        "--audio-format", "opus",
        "--audio-quality", "9",  # 最低质量
        "-o", raw_path + ".%(ext)s",
        "--no-playlist",
        "--proxy", os.environ.get("HTTP_PROXY", ""),
        url,
    ]
    subprocess.run(cmd_dl, check=True, capture_output=True, text=True)

    # 找到下载后的文件
    downloaded = None
    for f in Path(out_dir).glob("raw_audio.*"):
        downloaded = str(f)
        break
    if not downloaded:
        raise RuntimeError("yt-dlp 下载失败，未找到音频文件")

    # 用 ffmpeg 二次压缩：mono 16kHz，低码率 opus
    final_path = os.path.join(out_dir, "audio.ogg")
    cmd_ff = [
        "ffmpeg", "-y", "-i", downloaded,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "16k",
        final_path,
    ]
    subprocess.run(cmd_ff, check=True, capture_output=True, text=True)
    return final_path


def split_audio(audio_path: str, out_dir: str, chunk_sec: int = CHUNK_DURATION) -> list[str]:
    """如果文件超过 25MB，按时长分片."""
    size = os.path.getsize(audio_path)
    if size <= MAX_FILE_SIZE:
        return [audio_path]

    # 获取总时长
    probe = subprocess.run(
        ["ffmpeg", "-i", audio_path],
        capture_output=True, text=True,
    )
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", probe.stderr)
    if not duration_match:
        return [audio_path]
    h, m, s = map(int, duration_match.groups())
    total_sec = h * 3600 + m * 60 + s

    chunks = []
    start = 0
    idx = 0
    while start < total_sec:
        chunk_path = os.path.join(out_dir, f"chunk_{idx}.ogg")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(chunk_sec),
            "-c:a", "libopus", "-b:a", "16k",
            chunk_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        chunks.append(chunk_path)
        start += chunk_sec
        idx += 1
    return chunks


def seconds_to_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_to_srt(audio_path: str, time_offset: float = 0.0) -> list[dict]:
    """调用 Groq Whisper API 转录，返回 segments."""
    client = Groq(api_key=GROQ_API_KEY)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f),
            model="whisper-large-v3",
            language="zh",
            response_format="verbose_json",
        )

    segments = []
    if hasattr(result, "segments") and result.segments:
        for seg in result.segments:
            segments.append({
                "start": seg["start"] + time_offset,
                "end": seg["end"] + time_offset,
                "text": seg["text"].strip(),
            })
    elif hasattr(result, "text") and result.text:
        # fallback: 没有分段信息时整段返回
        segments.append({
            "start": time_offset,
            "end": time_offset + 30.0,
            "text": result.text.strip(),
        })
    return segments


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = seconds_to_srt_time(seg["start"])
        end_ts = seconds_to_srt_time(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "请输入视频链接"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. 下载音频
            audio_path = download_audio(url, tmp_dir)

            # 2. 分片（长视频）
            chunks = split_audio(audio_path, tmp_dir)

            # 3. 逐片转录
            all_segments = []
            for i, chunk in enumerate(chunks):
                offset = i * CHUNK_DURATION if len(chunks) > 1 else 0.0
                segs = transcribe_to_srt(chunk, time_offset=offset)
                all_segments.extend(segs)

            # 4. 生成 SRT
            srt_text = segments_to_srt(all_segments)
            return jsonify({"srt": srt_text, "segments": all_segments})

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr if e.stderr else str(e)
        return jsonify({"error": f"下载/转码失败: {err_msg}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5009)
