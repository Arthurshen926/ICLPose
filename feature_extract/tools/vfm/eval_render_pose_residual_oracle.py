"""Run the render-pose residual solver diagnostic on render-RGB matches.

This is a thin IF-C entry point. It delegates matching/rendering to
`eval_render_rgb_feature_keypoint_pose.py` and forces residual-solver fields
into the output rows and summary.
"""

from __future__ import annotations

import sys
from typing import Sequence

from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import main as _eval_render_rgb_main


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--enable_render_pose_residual_solver" not in args:
        args.append("--enable_render_pose_residual_solver")
    _eval_render_rgb_main(args)


if __name__ == "__main__":
    main()
