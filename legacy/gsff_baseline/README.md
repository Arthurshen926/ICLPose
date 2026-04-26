# GSFF baseline (legacy)

This directory contains the archived GSFF reproduction / comparison branch.

## Scope

This branch was kept to:

- reproduce GSFF-style feature-field localization
- compare GSFF against the current RADIO-based mainline
- preserve historical debugging / visualization scripts

It is **not** part of the current mainline.

## Layout

- `configs/` — GSFF experiment configs
- `scripts/` — GSFF train / eval / visualize entrypoints
- `docs/` — reproduction report and notes
- `reference/` — original paper artifacts copied into the repo

## Important note

The top-level `gsff/` Python package is still kept in-place so these archived scripts can import it, but it should be treated as a legacy comparison module rather than an actively developed mainline component.
