#!/usr/bin/env python3
"""Compatibility shim for the canonical PoseRefine pipeline evaluation entrypoint."""

from __future__ import annotations

from pose_refine.evaluate_pipeline import main


if __name__ == "__main__":
    main()
