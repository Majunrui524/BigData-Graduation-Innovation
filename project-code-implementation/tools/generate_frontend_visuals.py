#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import struct
import zlib
from pathlib import Path


def chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, pixels: list[tuple[int, int, int, int]]) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            r, g, b, a = pixels[y * width + x]
            raw.extend((r, g, b, a))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", header),
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            chunk(b"IEND", b""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def blend(base: tuple[int, int, int], overlay: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return tuple(
        max(0, min(255, round(base[channel] * (1.0 - alpha) + overlay[channel] * alpha)))
        for channel in range(3)
    )


def make_noise(width: int, height: int) -> list[tuple[int, int, int, int]]:
    rng = random.Random(42)
    base = (246, 239, 229)
    pixels = []
    for y in range(height):
        for x in range(width):
            delta = rng.randint(-10, 10)
            value = tuple(max(0, min(255, channel + delta)) for channel in base)
            if rng.random() < 0.005:
                value = blend(value, (168, 64, 45), 0.18)
            pixels.append((*value, 255))
    return pixels


def make_heatmap(width: int, height: int) -> list[tuple[int, int, int, int]]:
    base = (250, 244, 235)
    blobs = [
        (0.22, 0.25, 0.32, (168, 64, 45)),
        (0.72, 0.36, 0.28, (24, 59, 119)),
        (0.57, 0.72, 0.24, (178, 218, 65)),
        (0.34, 0.67, 0.18, (117, 43, 63)),
    ]
    pixels = []
    for y in range(height):
        for x in range(width):
            color = base
            nx = x / max(width - 1, 1)
            ny = y / max(height - 1, 1)
            for cx, cy, radius, blob_color in blobs:
                dist = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
                alpha = max(0.0, 1.0 - dist / radius) ** 2 * 0.72
                color = blend(color, blob_color, alpha)
            stripe = 0.035 * math.sin(nx * 11.0 + ny * 5.0)
            color = tuple(max(0, min(255, round(channel * (1.0 + stripe)))) for channel in color)
            pixels.append((*color, 255))
    return pixels


def make_poster(width: int, height: int) -> list[tuple[int, int, int, int]]:
    base = (245, 236, 223)
    pixels = []
    for y in range(height):
        for x in range(width):
            nx = x / max(width - 1, 1)
            ny = y / max(height - 1, 1)
            color = base
            if nx < 0.56:
                color = blend(color, (17, 17, 17), 0.08 + ny * 0.12)
            if 0.62 < nx < 0.92 and 0.1 < ny < 0.72:
                color = blend(color, (168, 64, 45), 0.35)
            if 0.08 < nx < 0.32 and 0.14 < ny < 0.82:
                color = blend(color, (24, 59, 119), 0.28)
            if ((nx - 0.72) ** 2 + (ny - 0.28) ** 2) < 0.045:
                color = blend(color, (178, 218, 65), 0.65)
            if ((nx - 0.25) ** 2 + (ny - 0.72) ** 2) < 0.03:
                color = blend(color, (117, 43, 63), 0.45)
            pixels.append((*color, 255))
    return pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raster visuals for the 10k frontend demo")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    write_png(output_dir / "paper-noise.png", 512, 512, make_noise(512, 512))
    write_png(output_dir / "hero-heatmap.png", 1600, 900, make_heatmap(1600, 900))
    write_png(output_dir / "community-risk-poster.png", 1200, 1600, make_poster(1200, 1600))
    print(output_dir)


if __name__ == "__main__":
    main()
