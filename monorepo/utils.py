import typing

import numpy as np
import datetime
import random
import math
import os
import cv2
import json
import hashlib

from PIL import Image


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
HASH_SHA256_LENGTH = 10

def erode_or_dilate(x, k):
    k = int(k)
    if k > 0:
        return cv2.dilate(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=k)
    if k < 0:
        return cv2.erode(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=-k)
    return x


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def apply_wildcards(wildcard_text, rng, i, read_wildcards_in_order):
    wildcards_max_bfs_depth = 64
    for _ in range(wildcards_max_bfs_depth):
        placeholders = re.findall(r'__([\w-]+)__', wildcard_text)
        if len(placeholders) == 0:
            return wildcard_text

        print(f'[Wildcards] processing: {wildcard_text}')
        for placeholder in placeholders:
            try:
                matches = [x for x in modules.config.wildcard_filenames if os.path.splitext(os.path.basename(x))[0] == placeholder]
                words = open(os.path.join(modules.config.path_wildcards, matches[0]), encoding='utf-8').read().splitlines()
                words = [x for x in words if x != '']
                assert len(words) > 0
                if read_wildcards_in_order:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', words[i % len(words)], 1)
                else:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
            except:
                print(f'[Wildcards] Warning: {placeholder}.txt missing or empty. '
                      f'Using "{placeholder}" as a normal word.')
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
            print(f'[Wildcards] {wildcard_text}')

    print(f'[Wildcards] BFS stack overflow. Current text: {wildcard_text}')
    return wildcard_text