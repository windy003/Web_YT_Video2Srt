import os
import re
import time
import uuid
import queue
import threading
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
# 本地上传音频文件大小上限
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Groq Whisper 单次上传限制 25MB
MAX_FILE_SIZE = 25 * 1024 * 1024
# 分片时长（秒），保证每片 < 25MB
CHUNK_DURATION = 600  # 10 分钟
# yt-dlp 超过这么久没有任何新输出，就判定为卡死并中止
STALL_TIMEOUT = 120

LATEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest.json")


def save_latest(data: dict) -> None:
    try:
        with open(LATEST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def load_latest() -> dict | None:
    try:
        with open(LATEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def download_audio(url: str, out_dir: str):
    """用 yt-dlp 下载最低码率音频，再用 ffmpeg 转为 mono 16kHz opus 进一步压缩.
    生成器：yield 速度信息字符串，最后 yield {'result': path}。
    """
    raw_path = os.path.join(out_dir, "raw_audio")
    proxy = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
    cmd_dl = [
        "yt-dlp",
        "--print", "before_dl:__TITLE__%(title)s",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "opus",
        "--audio-quality", "9",
        "--newline",
        "--remote-components", "ejs:github",
        "-o", raw_path + ".%(ext)s",
        "--no-playlist",
        url,
    ]
    if proxy:
        cmd_dl += ["--proxy", proxy]
    proc = subprocess.Popen(
        cmd_dl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **_subprocess_kwargs,
    )

    # 用后台线程读取 stdout，主循环通过队列 + 超时来检测"卡死无输出"
    line_queue: queue.Queue = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)  # EOF 哨兵

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    title = ""
    all_output = []
    while True:
        try:
            line = line_queue.get(timeout=STALL_TIMEOUT)
        except queue.Empty:
            proc.kill()
            raise RuntimeError(
                f"yt-dlp 下载已卡住超过 {STALL_TIMEOUT} 秒无任何响应，已自动中止。"
                f"可能原因：网络问题、代理配置不当，或 YouTube 触发了限流/反爬（可尝试更新 yt-dlp 或稍后重试）。"
            )
        if line is None:
            break
        line = line.strip()
        all_output.append(line)
        if line.startswith("__TITLE__"):
            title = line[len("__TITLE__"):].strip()
            yield {"title": title}
            continue
        # yt-dlp 进度行示例: [download]  45.0% of 1.23MiB at 500.00KiB/s ETA 00:02
        m = re.search(r'\[download\]\s+([\d.]+%)\s+of\s+\S+\s+at\s+(\S+/s)', line)
        if m and m.group(1) != '100.0%' and m.group(1) != '100%':
            yield {"speed": f"下载中 {m.group(1)} | {m.group(2)}"}
    proc.wait()
    if proc.returncode != 0:
        error_detail = "\n".join(all_output[-20:]) if all_output else "(无输出)"
        raise RuntimeError(f"yt-dlp 失败 (exit {proc.returncode}):\n{error_detail}")
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


class StageTracker:
    """记录各阶段耗时，并产出前端用的 SSE 'stage' 事件."""

    def __init__(self):
        self.label = None
        self.start = None
        self.history = []

    def send(self, stage):
        now = time.time()
        if self.label is not None and self.start is not None:
            self.history.append({"label": self.label, "seconds": int(now - self.start)})
        self.label = stage
        self.start = now
        yield f"data: {json.dumps({'type': 'stage', 'stage': stage})}\n\n"

    def finalize(self):
        if self.label is not None and self.start is not None:
            self.history.append({"label": self.label, "seconds": int(time.time() - self.start)})
            self.label = None
            self.start = None


def run_pipeline(tmp_dir: str, raw_path: str, video_title: str, tracker: StageTracker, source_label: str):
    """从已拿到的原始音频文件开始：压缩 -> 分片 -> 转录 -> 生成字幕.
    URL 下载和本地上传两条路径在拿到 raw_path 之后共用这段逻辑。
    """
    # 压缩音频（仅超过 25MB 时）
    if os.path.getsize(raw_path) > MAX_FILE_SIZE:
        yield from tracker.send("正在压缩音频")
        audio_path = compress_audio(raw_path, tmp_dir)
    else:
        audio_path = raw_path

    # 获取音频时长
    video_duration = get_audio_duration(audio_path)

    # 分片（长音频）
    yield from tracker.send("正在音频分片")
    chunks = split_audio(audio_path, tmp_dir)

    # 逐片转录
    all_segments = []
    model_name = "whisper-large-v3"
    for i, chunk in enumerate(chunks):
        chunk_label = f"分片 {i+1}/{len(chunks)}" if len(chunks) > 1 else ""

        # 上传阶段
        if chunk_label:
            yield from tracker.send(f"正在上传{chunk_label}到 Groq")
        else:
            yield from tracker.send("正在上传音频到 Groq")

        client = Groq(api_key=GROQ_API_KEY)
        with open(chunk, "rb") as f:
            file_tuple = (os.path.basename(chunk), f.read())

        # 转录阶段
        if chunk_label:
            yield from tracker.send(f"正在转录{chunk_label} | 模型: {model_name}")
        else:
            yield from tracker.send(f"正在转录音频 | 模型: {model_name}")

        result = client.audio.transcriptions.create(
            file=file_tuple,
            model=model_name,
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

    # 生成 SRT
    yield from tracker.send("正在生成字幕文件")
    srt_text = segments_to_srt(all_segments)
    tracker.finalize()
    save_latest({
        "url": source_label,
        "title": video_title,
        "srt": srt_text,
        "segments": all_segments,
        "duration": video_duration,
        "stages": tracker.history,
    })
    yield f"data: {json.dumps({'type': 'result', 'title': video_title, 'srt': srt_text, 'segments': all_segments, 'duration': video_duration})}\n\n"


@app.route("/")
def index():
    return render_template("index.html", latest=None)


@app.route("/latest.html")
def latest_page():
    return render_template("index.html", latest=load_latest())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "请输入视频链接"}), 400

    def generate():
        tracker = StageTracker()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 1. 下载音频
                yield from tracker.send("正在下载音频")
                raw_path = None
                video_title = ""
                for update in download_audio(url, tmp_dir):
                    if "title" in update:
                        video_title = update["title"]
                        yield f"data: {json.dumps({'type': 'title', 'title': video_title})}\n\n"
                    elif "speed" in update:
                        yield f"data: {json.dumps({'type': 'speed', 'text': update['speed']})}\n\n"
                    elif "result" in update:
                        raw_path = update["result"]

                yield from run_pipeline(tmp_dir, raw_path, video_title, tracker, url)

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr if e.stderr else str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': f'下载/转码失败: {err_msg}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route("/transcribe_upload", methods=["POST"])
def transcribe_upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "请选择要上传的音频文件"}), 400

    original_name = file.filename
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "文件为空"}), 400

    def generate():
        tracker = StageTracker()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 1. 保存上传的文件
                yield from tracker.send("正在保存上传的音频")
                ext = os.path.splitext(original_name)[1] or ".dat"
                raw_path = os.path.join(tmp_dir, "raw_audio" + ext)
                with open(raw_path, "wb") as f:
                    f.write(file_bytes)
                video_title = os.path.splitext(original_name)[0]
                yield f"data: {json.dumps({'type': 'title', 'title': video_title})}\n\n"

                yield from run_pipeline(tmp_dir, raw_path, video_title, tracker, f"[本地上传] {original_name}")

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr if e.stderr else str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': f'转码失败: {err_msg}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "文件过大，超过上传大小限制（1GB）"}), 413


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5009)
