import os
import re
import uuid
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response
import json
from groq import Groq

load_dotenv()

# Windows 下隐藏子进程窗口
_subprocess_kwargs = {}
if os.name == 'nt':
    _subprocess_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Groq Whisper 单次上传限制 25MB
MAX_FILE_SIZE = 25 * 1024 * 1024
# 分片时长（秒），保证每片 < 25MB
CHUNK_DURATION = 600  # 10 分钟


def download_audio(url: str, out_dir: str):
    """用 yt-dlp 下载最低码率音频，再用 ffmpeg 转为 mono 16kHz opus 进一步压缩.
    生成器：yield 速度信息字符串，最后 yield {'result': path}。
    """
    raw_path = os.path.join(out_dir, "raw_audio")
    cmd_dl = [
        "yt-dlp",
        "-f", "worstaudio/worst",
        "-x",
        "--audio-format", "opus",
        "--audio-quality", "9",
        "--newline",
        "-o", raw_path + ".%(ext)s",
        "--no-playlist",
        "--proxy", os.environ.get("HTTP_PROXY", ""),
        url,
    ]
    proc = subprocess.Popen(
        cmd_dl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **_subprocess_kwargs,
    )
    for line in proc.stdout:
        line = line.strip()
        # yt-dlp 进度行示例: [download]  45.0% of 1.23MiB at 500.00KiB/s ETA 00:02
        m = re.search(r'\[download\]\s+([\d.]+%)\s+of\s+\S+\s+at\s+(\S+/s)', line)
        if m and m.group(1) != '100.0%' and m.group(1) != '100%':
            yield {"speed": f"下载中 {m.group(1)} | {m.group(2)}"}
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd_dl)
    yield {"speed": ""}

    downloaded = None
    for f in Path(out_dir).glob("raw_audio.*"):
        downloaded = str(f)
        break
    if not downloaded:
        raise RuntimeError("yt-dlp 下载失败，未找到音频文件")
    yield {"result": downloaded}


def compress_audio(input_path: str, out_dir: str) -> str:
    """用 ffmpeg 压缩音频为 mono 16kHz opus."""
    final_path = os.path.join(out_dir, "audio.ogg")
    cmd_ff = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "16k",
        final_path,
    ]
    subprocess.run(cmd_ff, check=True, capture_output=True, text=True, **_subprocess_kwargs)
    return final_path


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）."""
    probe = subprocess.run(
        ["ffmpeg", "-i", audio_path],
        capture_output=True, text=True, **_subprocess_kwargs,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", probe.stderr)
    if not match:
        return 0.0
    h, m, s, cs = int(match[1]), int(match[2]), int(match[3]), int(match[4])
    return h * 3600 + m * 60 + s + cs / 100.0


def split_audio(audio_path: str, out_dir: str, chunk_sec: int = CHUNK_DURATION) -> list[str]:
    """如果文件超过 25MB，按时长分片."""
    size = os.path.getsize(audio_path)
    if size <= MAX_FILE_SIZE:
        return [audio_path]

    # 获取总时长
    probe = subprocess.run(
        ["ffmpeg", "-i", audio_path],
        capture_output=True, text=True, **_subprocess_kwargs,
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
        subprocess.run(cmd, check=True, capture_output=True, text=True, **_subprocess_kwargs)
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

    def generate():
        def send_stage(stage):
            yield f"data: {json.dumps({'type': 'stage', 'stage': stage})}\n\n"

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 1. 下载音频
                yield from send_stage("正在下载音频")
                raw_path = None
                for update in download_audio(url, tmp_dir):
                    if "speed" in update:
                        yield f"data: {json.dumps({'type': 'speed', 'text': update['speed']})}\n\n"
                    elif "result" in update:
                        raw_path = update["result"]

                # 2. 压缩音频（仅超过 25MB 时）
                if os.path.getsize(raw_path) > MAX_FILE_SIZE:
                    yield from send_stage("正在压缩音频")
                    audio_path = compress_audio(raw_path, tmp_dir)
                else:
                    audio_path = raw_path

                # 获取视频时长
                video_duration = get_audio_duration(audio_path)

                # 2. 分片（长视频）
                yield from send_stage("正在音频分片")
                chunks = split_audio(audio_path, tmp_dir)

                # 3. 逐片转录
                all_segments = []
                model_name = "whisper-large-v3"
                for i, chunk in enumerate(chunks):
                    chunk_label = f"分片 {i+1}/{len(chunks)}" if len(chunks) > 1 else ""

                    # 上传阶段
                    if chunk_label:
                        yield from send_stage(f"正在上传{chunk_label}到 Groq")
                    else:
                        yield from send_stage("正在上传音频到 Groq")

                    client = Groq(api_key=GROQ_API_KEY)
                    with open(chunk, "rb") as f:
                        file_tuple = (os.path.basename(chunk), f.read())

                    # 转录阶段
                    if chunk_label:
                        yield from send_stage(f"正在转录{chunk_label} | 模型: {model_name}")
                    else:
                        yield from send_stage(f"正在转录音频 | 模型: {model_name}")

                    result = client.audio.transcriptions.create(
                        file=file_tuple,
                        model=model_name,
                        language="zh",
                        response_format="verbose_json",
                    )

                    offset = i * CHUNK_DURATION if len(chunks) > 1 else 0.0
                    if hasattr(result, "segments") and result.segments:
                        for seg in result.segments:
                            all_segments.append({
                                "start": seg["start"] + offset,
                                "end": seg["end"] + offset,
                                "text": seg["text"].strip(),
                            })
                    elif hasattr(result, "text") and result.text:
                        all_segments.append({
                            "start": offset,
                            "end": offset + 30.0,
                            "text": result.text.strip(),
                        })

                # 4. 生成 SRT
                yield from send_stage("正在生成字幕文件")
                srt_text = segments_to_srt(all_segments)
                yield f"data: {json.dumps({'type': 'result', 'srt': srt_text, 'segments': all_segments, 'duration': video_duration})}\n\n"

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr if e.stderr else str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': f'下载/转码失败: {err_msg}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5009)
