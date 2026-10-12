"""Media facts: header probes and the site census. ``[media]`` extra; no protocol semantics.

``probe_info(file, kind)`` reads a stored file's header (PyAV for video and container audio, soundfile
for wav/flac, pillow for images; the ``ffprobe`` CLI when that backend is absent) and returns the ``MediaInfo`` dict: every key of
``yavih.sample.MEDIA_INFO_KEYS``, ``None`` where the header does not say or the key does not apply to the
kind. Nothing is decoded: with ``demux=True`` a fact the header lacks is taken from the demuxed packets
(``n_frames`` for webm, whose header carries no frame count; ``n_samples`` for container audio without a
stream duration). ``has_track`` (video, audio) says whether the file holds a track of that kind: a readable
container without one (an mp4 with a video track and no audio track) is a fact, not an error — ``has_track:
False`` with the track's facts null and ``container`` / ``bytes`` set.

``write_census(bench)`` probes every present media cell of a benchmark (``demux=True``) and writes the
media-info table ``$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz`` the reader prefers over a live
probe (``yavih bench verify --census``). Memory stays flat in the number of cells: a stored file is probed by
its path (never read into memory), a tar member's bytes are dropped once its probe is done, at most a few cells
per worker are in flight, and a progress line goes to stderr every ``progress_every`` cells. Without a census
row the reader falls back to a live header probe of the same file: slower, and exactly the same facts.
"""

from __future__ import annotations

import io
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, BinaryIO, TextIO

from yavih.media import DecodeError, DecoderMissing

__all__ = ["probe_info", "write_census", "container_name", "header_backends", "missing_header_backends"]

_FORMAT_CANON = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "wav": "wav", "flac": "flac", "mp4": "mp4", "webm": "webm", "mkv": "mkv", "m4a": "m4a", "mov": "mov", "avi": "avi", "mp3": "mp3", "ogg": "ogg"}
_FORMAT_FIRST = {"mov,mp4,m4a,3gp,3g2,mj2": "mp4", "matroska,webm": "webm"}


def container_name(format_names: str | None, ext: str | None = None) -> str | None:
    """Canonical container name: the file extension when the demuxer accepts it (``mp4`` vs ``mov`` share one
    demuxer), else the demuxer's usual name (``mov,mp4,…`` → ``mp4``, ``matroska,webm`` → ``webm``)."""
    ext = (ext or "").lower().lstrip(".") or None
    names = [n.strip().lower() for n in (format_names or "").split(",") if n.strip()]
    if ext is not None:
        canon = _FORMAT_CANON.get(ext, ext)
        demuxer = {"mkv": "matroska", "jpeg": "jpeg_pipe", "png": "png_pipe"}.get(canon)
        if not names or ext in names or canon in names or demuxer in names or (canon == "jpeg" and "mjpeg" in names):
            return canon
    if format_names in _FORMAT_FIRST:
        return _FORMAT_FIRST[format_names]
    if names:
        n = names[0]
        return {"jpeg_pipe": "jpeg", "png_pipe": "png", "mjpeg": "jpeg"}.get(n, _FORMAT_CANON.get(n, n))
    return None


def _empty() -> dict[str, Any]:
    from yavih.sample import MEDIA_INFO_KEYS

    return {k: None for k in MEDIA_INFO_KEYS}


def _as_file(file: Any) -> tuple[Any, str | None, int | None]:
    """``(source for the decoder, extension hint, size in bytes)``. A ``RawStream`` over a stored file whose bytes
    are not cached yet is probed BY PATH (the header is read in place; nothing is loaded or cached), so probing
    many streams keeps no file in memory; a tar member, cached bytes or an absent file go through ``bytes()``
    (an absent file raises ``MediaMissing`` there)."""
    if hasattr(file, "kind") and hasattr(file, "bytes") and hasattr(file, "open"):  # a RawStream
        real = getattr(file, "_real_path", None)
        key = real or (file.member[0] if file.member else None)  # for the extension only
        ext = key.rsplit(".", 1)[-1].lower() if key and "." in key.rsplit("/", 1)[-1] else None
        ext = getattr(file, "_ext", None) or ext
        src = getattr(file, "_src", None)
        if real is not None and not getattr(src, "cached", False):
            size = file.nbytes
            if size is not None and os.path.isfile(real):
                return os.path.abspath(real), ext, int(size)
        data = file.bytes()
        return io.BytesIO(data), ext, len(data)
    if isinstance(file, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(file)), None, len(file)
    if isinstance(file, (str, os.PathLike)):
        p = os.fspath(file)
        base = p.rsplit("/", 1)[-1]
        return p, (base.rsplit(".", 1)[-1].lower() if "." in base else None), os.path.getsize(p)
    if hasattr(file, "read"):
        size = None
        try:
            pos = file.tell()
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(pos)
        except (OSError, AttributeError):
            pass
        return file, None, size
    raise TypeError(f"probe_info: unsupported file {type(file).__name__}")


def _rewind(src: Any) -> Any:
    if hasattr(src, "seek"):
        src.seek(0)
    return src


def _av():
    try:
        import av
    except ImportError as exc:
        raise DecoderMissing("av") from exc
    return av


def _rate(stream: Any) -> float | None:
    for attr in ("average_rate", "guessed_rate", "base_rate"):
        r = getattr(stream, attr, None)
        if r:
            try:
                f = float(r)
            except (TypeError, ZeroDivisionError):
                continue
            if f > 0:
                return f
    return None


def _video(src: Any, ext: str | None, demux: bool) -> dict[str, Any]:
    av = _av()
    out: dict[str, Any] = {}
    try:
        with av.open(_rewind(src), mode="r") as c:
            if not c.streams.video:  # a readable file without a video track: a fact, not an error
                return {"has_track": False, "container": container_name(c.format.name, ext)}
            out["has_track"] = True
            vs = c.streams.video[0]
            if vs.duration is not None and vs.time_base is not None:
                out["duration_s"] = float(vs.duration * vs.time_base)
            elif c.duration is not None:
                out["duration_s"] = c.duration / 1_000_000.0
            out["fps"] = _rate(vs)
            out["n_frames"] = int(vs.frames) if vs.frames else None
            out["width"] = int(vs.codec_context.width) or None
            out["height"] = int(vs.codec_context.height) or None
            rot = (vs.metadata or {}).get("rotate")
            if rot not in (None, "", "0"):
                try:
                    out["rotation"] = int(float(rot))
                except ValueError:
                    pass
            out["container"] = container_name(c.format.name, ext)
            if demux and not out["n_frames"]:  # webm headers carry no frame count: count packets, decode nothing
                n = 0
                for pkt in c.demux(vs):
                    if pkt.size:  # the trailing flush packet is empty
                        n += 1
                out["n_frames"] = n or None
    except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
        raise DecodeError(f"PyAV could not read the video header: {exc}") from exc
    return out


def _audio_av(src: Any, ext: str | None, demux: bool) -> dict[str, Any]:
    av = _av()
    out: dict[str, Any] = {}
    try:
        with av.open(_rewind(src), mode="r") as c:
            if not c.streams.audio:  # a readable file without an audio track: a fact, not an error
                return {"has_track": False, "container": container_name(c.format.name, ext)}
            out["has_track"] = True
            st = c.streams.audio[0]
            sr = int(st.codec_context.sample_rate or st.rate or 0) or None
            out["sample_rate"] = sr
            out["channels"] = int(st.codec_context.channels or 0) or None
            out["container"] = container_name(c.format.name, ext)
            ns = None
            if sr and st.duration is not None and st.time_base is not None:
                # the header's duration honours the edit list (AAC priming excluded); packet sums do not
                ns = int(round(float(st.duration * st.time_base) * sr))
            if ns is None and demux and sr:
                total = 0
                for pkt in c.demux(st):
                    if pkt.size and pkt.duration:
                        total += int(pkt.duration)
                if total and st.time_base is not None:
                    ns = int(round(float(total * st.time_base) * sr))
            out["n_samples"] = ns
            if ns is not None and sr:
                out["duration_s"] = ns / float(sr)
            elif c.duration is not None:
                out["duration_s"] = c.duration / 1_000_000.0
    except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
        raise DecodeError(f"PyAV could not read the audio header: {exc}") from exc
    return out


def _audio(src: Any, ext: str | None, demux: bool) -> dict[str, Any]:
    try:
        import soundfile as sf
    except (ImportError, OSError):
        sf = None
    if sf is not None:
        try:
            i = sf.info(_rewind(src))
            fmt = str(i.format or "").lower()
            return {
                "sample_rate": int(i.samplerate),
                "n_samples": int(i.frames),
                "channels": int(i.channels),
                "duration_s": int(i.frames) / float(i.samplerate),
                "container": container_name(fmt, ext),
                "has_track": True,
            }
        except Exception:  # noqa: BLE001 - libsndfile does not read mp4/aac; PyAV does
            pass
    try:
        return _audio_av(src, ext, demux)
    except DecoderMissing:
        if sf is None:
            raise DecoderMissing("soundfile", "pip install 'yavih[media]' (soundfile for wav/flac, av for container audio)") from None
        raise


def _image(src: Any, ext: str | None) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DecoderMissing("pillow") from exc
    try:
        with Image.open(_rewind(src)) as im:  # header only; no pixel decode
            w, h = im.size
            fmt = (im.format or "").lower()
    except (OSError, SyntaxError, ValueError) as exc:
        raise DecodeError(f"pillow could not read the image header: {exc}") from exc
    return {"width": int(w), "height": int(h), "container": container_name(fmt, ext)}


# ----------------------------------------------------------------------------- ffprobe fallback


#: Python header backends per stream kind, in the order ``probe_info`` tries them.
_PY_BACKENDS: dict[str, tuple[str, ...]] = {"video": ("av",), "audio": ("soundfile", "av"), "image": ("pillow",)}
_IMPORT_NAME = {"av": "av", "soundfile": "soundfile", "pillow": "PIL.Image"}


def _importable(name: str) -> bool:
    try:
        __import__(_IMPORT_NAME.get(name, name))
    except (ImportError, OSError):  # soundfile raises OSError when libsndfile is absent
        return False
    return True


def _ffprobe_available() -> bool:
    import shutil

    return shutil.which("ffprobe") is not None


def header_backends(kind: str) -> list[str]:
    """The header backends ``probe_info`` can use for stream ``kind`` on this host: the importable Python ones,
    then ``ffprobe`` when it is on ``PATH``. Empty = no fact can be read live."""
    out = [b for b in _PY_BACKENDS.get(kind, ()) if _importable(b)]
    if _ffprobe_available():
        out.append("ffprobe")
    return out


def missing_header_backends(kinds: Any) -> list[str]:
    """The stream kinds among ``kinds`` (``video | audio | image``) for which no header backend exists here,
    each once, in order: without a census every row of such a stream would fail ``DecoderMissing``."""
    out: list[str] = []
    for k in kinds:
        if k not in out and not header_backends(k):
            out.append(k)
    return out


def _frac(s: Any) -> float | None:
    if s in (None, "", "N/A", "0/0"):
        return None
    try:
        if isinstance(s, str) and "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) if float(b) else None
        return float(s)
    except (TypeError, ValueError):
        return None


def _ffprobe_json(src: Any, *, count_packets: bool) -> dict[str, Any]:
    """``ffprobe -show_streams -show_format -of json`` of a path, or of bytes / a file object (via stdin; through a
    temporary file when packets must be counted, which needs a seekable input)."""
    import json
    import subprocess

    base = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_packets:
        base.insert(3, "-count_packets")
    tmp = None
    try:
        if isinstance(src, str):
            cmd, data = [*base, src], None
        else:
            data = _rewind(src).read()
            if count_packets:
                import tempfile

                fd, tmp = tempfile.mkstemp(prefix="yavih-ffprobe-")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                cmd, data = [*base, tmp], None
            else:
                cmd = [*base, "-i", "pipe:0"]
        try:
            proc = subprocess.run(cmd, input=data, capture_output=True, check=False, timeout=600)
        except OSError as exc:
            raise DecoderMissing("ffmpeg") from exc
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if proc.returncode != 0:
        raise DecodeError(f"ffprobe could not read the header: {proc.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise DecodeError(f"ffprobe output is not JSON: {exc}") from exc


def _ff_stream_duration(st: Mapping[str, Any]) -> float | None:
    """Stream duration in seconds: ``duration_ts * time_base`` (what PyAV's ``stream.duration * time_base`` gives)."""
    ts, tb = st.get("duration_ts"), _frac(st.get("time_base"))
    if isinstance(ts, int) and tb:
        return float(ts * tb)
    return _frac(st.get("duration"))


def _ffprobe(src: Any, kind: str, ext: str | None, demux: bool) -> dict[str, Any]:
    """The facts of ``_video`` / ``_audio_av`` / ``_image`` read with the ffprobe CLI (same libav, same semantics)."""
    j = _ffprobe_json(src, count_packets=demux and kind == "video")
    fmt = j.get("format") or {}
    container = container_name(fmt.get("format_name"), ext)
    if kind == "image":
        vs = next((s for s in j.get("streams") or [] if s.get("codec_type") == "video"), None)
        if vs is None or not vs.get("width"):
            raise DecodeError("ffprobe found no image in the file")
        return {"width": int(vs["width"]), "height": int(vs["height"]), "container": container}
    codec = "video" if kind == "video" else "audio"
    st = next((s for s in j.get("streams") or [] if s.get("codec_type") == codec), None)
    if st is None:  # a readable file without such a track: a fact, not an error
        return {"has_track": False, "container": container}
    out: dict[str, Any] = {"has_track": True, "container": container}
    fmt_dur = _frac(fmt.get("duration"))
    if kind == "video":
        d = _ff_stream_duration(st)
        out["duration_s"] = d if d is not None else fmt_dur
        out["fps"] = _frac(st.get("avg_frame_rate")) or _frac(st.get("r_frame_rate"))
        nb = st.get("nb_frames")
        out["n_frames"] = int(nb) if nb not in (None, "", "N/A") and int(nb) > 0 else None
        if demux and not out["n_frames"]:
            npk = st.get("nb_read_packets")
            out["n_frames"] = int(npk) if npk not in (None, "", "N/A") and int(npk) > 0 else None
        out["width"] = int(st.get("width") or 0) or None
        out["height"] = int(st.get("height") or 0) or None
        rot = (st.get("tags") or {}).get("rotate")
        if rot not in (None, "", "0"):
            try:
                out["rotation"] = int(float(rot))
            except ValueError:
                pass
        return out
    sr = int(_frac(st.get("sample_rate")) or 0) or None
    out["sample_rate"] = sr
    out["channels"] = int(st.get("channels") or 0) or None
    d = _ff_stream_duration(st)
    ns = int(round(d * sr)) if sr and d is not None else None
    out["n_samples"] = ns
    if ns is not None and sr:
        out["duration_s"] = ns / float(sr)
    else:
        out["duration_s"] = fmt_dur
    return out


def probe_info(file: str | os.PathLike | bytes | BinaryIO | Any, kind: str, *, demux: bool = False, ext: str | None = None) -> dict[str, Any]:
    """The ``MediaInfo`` of one stored file for stream ``kind`` (``video | audio | image``; a ``frames`` set's
    facts are assembled by the reader from its stored frames). ``file``: a path, bytes, a binary file object or
    a ``RawStream``. ``DecoderMissing`` when no backend is available, ``DecodeError`` for an unreadable header. A
    readable container without a track of ``kind`` gives ``has_track: False`` (track facts null), never an error.

    When the Python backend is absent (PyAV / soundfile / pillow), the ``ffprobe`` CLI reads the same header facts
    (same libav, same rules), so a model env without ``yavih[media]`` still reads media without a census."""
    if kind not in ("video", "audio", "image"):
        raise ValueError(f"probe_info: kind must be video, audio or image, got {kind!r}")
    src, ext_hint, size = _as_file(file)
    ext = ext or ext_hint
    info = _empty()
    try:
        if kind == "video":
            info.update(_video(src, ext, demux))
        elif kind == "audio":
            info.update(_audio(src, ext, demux))
        else:
            info.update(_image(src, ext))
    except DecoderMissing:
        if not _ffprobe_available():
            raise
        info.update(_ffprobe(src, kind, ext, demux))
    info["bytes"] = size
    return info


def _release(st: Any) -> None:
    """Drop the bytes a probe cached on ``st``'s byte source (``write_census`` holds every stream until the
    table is written; see ``yavih.sample._Source.release``)."""
    rel = getattr(getattr(st, "_src", None), "release", None)
    if rel is not None:
        rel()


def write_census(
    bench: Any,
    *,
    workers: int | None = None,
    path: str | os.PathLike | None = None,
    progress_every: int = 1000,
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """Probe every present distinct ``(stream, media cell)`` of ``bench`` (read splits; ``demux=True``) and
    write the media-info table. Returns ``{rows, sha256, path, missing, failed, failures}``.
    Missing media are skipped (they stay missing rows); an unreadable header counts as ``failed``.

    Memory is flat in the number of cells: stored files are probed by path, a tar member's cached bytes are dropped
    after its probe, and at most ``4 x workers`` cells are in flight. With ``progress_every > 0`` and at least
    that many cells, a progress line goes to ``progress_file`` (default stderr) every ``progress_every`` cells
    and at least once a minute."""
    from yavih.media import MediaMissing
    from yavih.sample import (
        MediaInfoTable,
        _SplitCtx,
        build_sample,
        default_decode_workers,
        media_info_path,
    )

    roots = bench.resolved_roots()
    todo: dict[tuple[str, str], Any] = {}
    for split in bench.spec.splits.read_values():
        man = bench.manifest(split)
        ctx = _SplitCtx(bench, split, roots, None)
        for i in range(len(man)):
            s = build_sample(bench, split, man, i, ctx=ctx)
            for name, st in s.streams.items():
                key = (name, man.media[i][name])
                todo.setdefault(key, st)

    def _probe_one(item: tuple[tuple[str, str], Any]) -> tuple[tuple[str, str], dict | None, str | None]:
        key, st = item
        try:
            return _probe_cell(key, st)
        finally:
            _release(st)

    def _probe_cell(key: tuple[str, str], st: Any) -> tuple[tuple[str, str], dict | None, str | None]:
        try:
            return key, probe_info(st, st.kind, demux=True), None
        except MediaMissing:
            return key, None, "missing"
        except DecodeError as exc:
            return key, None, f"{type(exc).__name__}: {exc}"

    entries: dict[tuple[str, str], Mapping[str, Any]] = {}
    missing = failed = 0
    failures: list[dict[str, str]] = []
    n_workers = max(1, int(workers or default_decode_workers()))
    n_cells = len(todo)
    report = progress_every > 0 and n_cells >= progress_every
    out_fh = progress_file if progress_file is not None else sys.stderr
    t0 = last = time.monotonic()
    n_done = 0

    def _progress(final: bool = False) -> None:
        wall = time.monotonic() - t0
        rate = n_done / wall if wall > 0 else 0.0
        eta = f", eta {(n_cells - n_done) / rate / 60:.1f} min" if rate > 0 and not final else ""
        word = "done" if final else "probed"
        print(f"[yavih] census {bench.name}: {n_done}/{n_cells} cells {word} ({missing} missing, {failed} failed), "
              f"{rate:.1f} cells/s{eta}", file=out_fh, flush=True)

    items = iter(todo.items())
    pending: set[Future] = set()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:

        def _fill() -> None:
            while len(pending) < 4 * n_workers:
                item = next(items, None)
                if item is None:
                    return
                pending.add(pool.submit(_probe_one, item))

        _fill()
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                key, info, err = fut.result()
                if info is not None:
                    entries[key] = info
                elif err == "missing":
                    missing += 1
                else:
                    failed += 1
                    failures.append({"stream": key[0], "key": key[1], "error": str(err)})
                n_done += 1
                now = time.monotonic()
                if report and (n_done % progress_every == 0 or now - last >= 60.0) and n_done < n_cells:
                    last = now
                    _progress()
            _fill()
    if report:
        _progress(final=True)
    failures = sorted(failures, key=lambda f: (f["stream"], f["key"]))[:10]  # deterministic, whatever the finish order
    out = path or media_info_path(bench.name, bench.manifest_hash)
    table = MediaInfoTable.write(out, entries)
    return {"rows": len(table), "sha256": table.sha256, "path": table.path, "missing": missing, "failed": failed, "failures": failures}
