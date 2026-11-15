# Copyright (c) 2026 Jintao Li.
# Zhejiang University (ZJU).
#
# Licensed under the MIT License.

"""Viewport-local cache for filtered windows and replay checkpoints."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from filark.utils.perf import elapsed_ms, log_perf, perf_debug_enabled, perf_now


class StatefulReplayCache2D:
    """Viewport-scoped replay cache for stateful streaming post-processing."""

    def __init__(
        self,
        *,
        source,
        image,
        buffer_h: int,
        buffer_w: int,
        window_cache_blocks: int = 4,
        checkpoint_stride: int = 256,
        checkpoint_limit: int = 32,
    ) -> None:
        self.source = source
        self.image = image
        self.buffer_h = int(buffer_h)
        self.buffer_w = int(buffer_w)

        self._window_cache_blocks = int(max(0, window_cache_blocks))
        self._checkpoint_stride = int(max(1, checkpoint_stride))
        self._checkpoint_limit = int(max(0, checkpoint_limit))

        self._window_cache: OrderedDict[tuple, tuple[np.ndarray, object]] = OrderedDict()
        self._checkpoints: OrderedDict[tuple, object] = OrderedDict()

    def has_stateful_processing(self) -> bool:
        fn = getattr(self.source, "has_stateful_processing", None)
        return bool(callable(fn) and fn())

    def can_snapshot_processing(self) -> bool:
        fn = getattr(self.source, "can_snapshot_processing_state", None)
        return bool(callable(fn) and fn())

    @staticmethod
    def _cache_scale_token(value: float) -> float:
        return round(float(value), 6)

    @staticmethod
    def _source_transform_token(source) -> tuple[str | None, int]:
        target = getattr(source, "data", source)
        key = getattr(target, "transform", None)
        if callable(key):
            key = key()
        rev = getattr(target, "transform_revision", 0)
        if callable(rev):
            rev = rev()
        try:
            rev_int = int(rev)
        except Exception:
            rev_int = 0
        return (None if key is None else str(key), rev_int)

    def _cache_scope(self, c0=None, h=None, w=None):
        h = self.buffer_h if h is None else int(h)
        w = self.buffer_w if w is None else int(w)
        transform_key, transform_revision = self._source_transform_token(self.source)
        return (
            id(getattr(self.source, "data", None)),
            transform_key,
            transform_revision,
            int(0 if c0 is None else c0),
            h,
            w,
            self._cache_scale_token(getattr(self.source, "scale_x", 1.0)),
            self._cache_scale_token(getattr(self.source, "scale_y", 1.0)),
            str(getattr(self.source, "y_mode", "local")),
            str(getattr(self.source, "downsample_mode", "drop")),
            id(getattr(self.source, "preprocess", None)),
            id(getattr(self.source, "postprocess", None)),
        )

    def _window_cache_key(self, t0, c0=None, h=None, w=None):
        return (self._cache_scope(c0=c0, h=h, w=w), int(t0))

    def _checkpoint_key(self, x_pos, c0=None, h=None, w=None):
        return (self._cache_scope(c0=c0, h=h, w=w), int(x_pos))

    def _cache_window(self, block: np.ndarray, t0: int, c0: int, *, end_state) -> None:
        if self._window_cache_blocks <= 0:
            return
        key = self._window_cache_key(t0, c0)
        self._window_cache.pop(key, None)
        self._window_cache[key] = (np.array(block, copy=True, order="C"), end_state)
        while len(self._window_cache) > self._window_cache_blocks:
            self._window_cache.popitem(last=False)

    def _get_cached_window(self, t0: int, c0: int):
        key = self._window_cache_key(t0, c0)
        hit = self._window_cache.get(key)
        if hit is None:
            return None
        self._window_cache.move_to_end(key)
        return hit

    def _store_checkpoint(self, x_pos: int, c0: int, *, state) -> None:
        if self._checkpoint_limit <= 0:
            return
        key = self._checkpoint_key(x_pos, c0)
        self._checkpoints.pop(key, None)
        self._checkpoints[key] = state
        while len(self._checkpoints) > self._checkpoint_limit:
            self._checkpoints.popitem(last=False)

    def _find_checkpoint(self, t0: int, c0: int):
        scope = self._cache_scope(c0=c0)
        best = None
        best_x = None
        for (key_scope, x_pos), state in self._checkpoints.items():
            if key_scope != scope or x_pos > int(t0):
                continue
            if best_x is None or x_pos > best_x:
                best_x = x_pos
                best = state
        if best_x is None:
            return None
        key = self._checkpoint_key(best_x, c0)
        self._checkpoints.move_to_end(key)
        return int(best_x), best

    def _capture_current_window(self) -> np.ndarray | None:
        snap = getattr(self.image, "snapshot_logical", None)
        if not callable(snap):
            return None
        block = snap()
        if block is None:
            return None
        if block.shape != (self.buffer_h, self.buffer_w):
            return None
        return block

    def _max_replay_width(self, visible_w: int) -> tuple[int, int]:
        warmup = 0
        fn = getattr(self.source, "processing_warmup_width", None)
        if callable(fn):
            try:
                warmup = int(max(0, fn(int(visible_w))))
            except Exception:
                warmup = 0
        if warmup <= 0:
            warmup = int(max(0, visible_w))
        # If replay spans further than a fresh warmup would need, discard state
        # and let StreamingSource.read_with_history() rebuild from a clean reset.
        limit = int(max(visible_w, visible_w + warmup))
        return limit, warmup

    def maybe_store_state_anchor(self, *, t0: int, c0: int, block: np.ndarray | None = None) -> None:
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        if (not self.has_stateful_processing()) or (not self.can_snapshot_processing()):
            return

        capture = getattr(self.source, "capture_processing_state", None)
        if not callable(capture):
            return
        state = capture()
        if state is None:
            return

        end_x = int(t0 + self.buffer_w)
        scope = self._cache_scope(c0=c0)
        best_x = None
        for (key_scope, x_pos) in self._checkpoints.keys():
            if key_scope != scope:
                continue
            if best_x is None or x_pos > best_x:
                best_x = x_pos
        should_store = (best_x is None) or (end_x - best_x >= self._checkpoint_stride)
        if should_store:
            self._store_checkpoint(end_x, c0, state=state)
            if block is None:
                block = self._capture_current_window()
            if block is not None:
                self._cache_window(block, t0, c0, end_state=state)
        if perf_on:
            log_perf(
                "replay_cache.store_anchor",
                t0=int(t0),
                c0=int(c0),
                end_x=int(end_x),
                stored=int(should_store),
                has_block=int(block is not None),
                checkpoint_count=int(len(self._checkpoints)),
                window_cache_count=int(len(self._window_cache)),
                ms=elapsed_ms(total_start),
            )

    def read_window(self, *, c0: int, t0: int, h=None, w=None, debug_context: dict | None = None):
        perf_on = perf_debug_enabled()
        total_start = perf_now() if perf_on else 0.0
        h = self.buffer_h if h is None else int(h)
        w = self.buffer_w if w is None else int(w)
        if (not self.has_stateful_processing()) or (not self.can_snapshot_processing()):
            return None

        restore = getattr(self.source, "restore_processing_state", None)
        if not callable(restore):
            return None

        cached = self._get_cached_window(int(t0), int(c0))
        if cached is not None:
            block, end_state = cached
            restore(end_state)
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    cache_mode="window_hit",
                    t0=int(t0),
                    c0=int(c0),
                    h=int(h),
                    w=int(w),
                    ms=elapsed_ms(total_start),
                )
                log_perf("replay_cache.read_window", **fields)
            return np.array(block, copy=True, order="C")

        hit = self._find_checkpoint(int(t0), int(c0))
        if hit is None:
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    cache_mode="miss",
                    t0=int(t0),
                    c0=int(c0),
                    h=int(h),
                    w=int(w),
                    ms=elapsed_ms(total_start),
                )
                log_perf("replay_cache.read_window", **fields)
            return None

        cp_x, cp_state = hit
        total_w = int(t0) - int(cp_x) + w
        if total_w <= 0:
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    cache_mode="invalid_checkpoint",
                    t0=int(t0),
                    c0=int(c0),
                    cp_x=int(cp_x),
                    h=int(h),
                    w=int(w),
                    total_w=int(total_w),
                    ms=elapsed_ms(total_start),
                )
                log_perf("replay_cache.read_window", **fields)
            return None

        replay_limit, warmup_w = self._max_replay_width(w)
        if total_w > replay_limit:
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    cache_mode="checkpoint_too_far",
                    t0=int(t0),
                    c0=int(c0),
                    cp_x=int(cp_x),
                    h=int(h),
                    w=int(w),
                    total_w=int(total_w),
                    replay_limit=int(replay_limit),
                    warmup_w=int(warmup_w),
                    ms=elapsed_ms(total_start),
                )
                log_perf("replay_cache.read_window", **fields)
            return None

        restore_start = perf_now() if perf_on else 0.0
        restore(cp_state)
        restore_ms = elapsed_ms(restore_start) if perf_on else 0.0
        out = self.source.read(
            int(c0),
            int(cp_x),
            h,
            total_w,
            debug_context=dict(debug_context or {}, cache_mode="checkpoint_replay", checkpoint_x=int(cp_x)),
        )
        if out is None:
            if perf_on:
                fields = dict(debug_context or {})
                fields.update(
                    cache_mode="checkpoint_none",
                    t0=int(t0),
                    c0=int(c0),
                    cp_x=int(cp_x),
                    h=int(h),
                    w=int(w),
                    total_w=int(total_w),
                    restore_ms=float(restore_ms),
                    ms=elapsed_ms(total_start),
                )
                log_perf("replay_cache.read_window", **fields)
            return None

        offset = int(t0) - int(cp_x)
        block = out[:, offset:offset + w]
        if block.shape[1] != w:
            block = np.pad(
                block,
                pad_width=((0, 0), (0, max(0, w - block.shape[1]))),
                mode="constant",
                constant_values=0.0,
            )
        if perf_on:
            fields = dict(debug_context or {})
            fields.update(
                cache_mode="checkpoint_hit",
                t0=int(t0),
                c0=int(c0),
                cp_x=int(cp_x),
                h=int(h),
                w=int(w),
                total_w=int(total_w),
                replay_limit=int(replay_limit),
                warmup_w=int(warmup_w),
                offset=int(offset),
                restore_ms=float(restore_ms),
                ms=elapsed_ms(total_start),
            )
            log_perf("replay_cache.read_window", **fields)
        return np.array(block, copy=True, order="C")
