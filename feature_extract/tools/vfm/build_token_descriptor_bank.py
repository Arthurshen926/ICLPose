"""Build a lightweight descriptor bank from a dense token manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    apply_pca_whitening_transform,
    build_token_descriptor_bank,
    fit_pca_whitening_transform,
    fit_vlad_codebook_from_manifest,
    fit_vlad_codebook_from_manifests,
    load_pca_whitening_transform_npz,
    load_vlad_codebook_npz,
    save_pca_whitening_transform_npz,
    save_vlad_codebook_npz,
)
from feature_extract.vfm.tokens import TokenBankManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pooled token descriptors")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--pooling", default="mean", choices=["mean", "gem", "vlad"])
    parser.add_argument("--gem_power", type=float, default=3.0)
    parser.add_argument("--vlad_clusters", type=int, default=32)
    parser.add_argument("--vlad_iterations", type=int, default=20)
    parser.add_argument("--vlad_max_tokens", type=int, default=200000)
    parser.add_argument("--vlad_codebook_tokens_per_image", type=int, default=0)
    parser.add_argument("--vlad_codebook_max_images", type=int, default=0)
    parser.add_argument("--vlad_codebook_manifest", action="append", default=[])
    parser.add_argument("--vlad_codebook_input", default="")
    parser.add_argument("--vlad_codebook_output", default="")
    parser.add_argument("--vlad_tokens_per_image", type=int, default=0)
    parser.add_argument("--descriptor_power", type=float, default=1.0)
    parser.add_argument("--pca_whitening_input", default="")
    parser.add_argument("--pca_whitening_output", default="")
    parser.add_argument("--pca_dim", type=int, default=0)
    parser.add_argument("--pca_whitening_epsilon", type=float, default=1e-6)
    parser.add_argument("--normalize_tokens", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metadata = {}
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text())
    manifest = TokenBankManifest.from_json(Path(args.manifest))
    metadata = {
        **metadata,
        "token_manifest": str(args.manifest),
        "token_manifest_sha256": file_sha256_short(Path(args.manifest)),
    }
    vlad_codebook = None
    if str(args.pooling) == "vlad":
        if str(args.vlad_codebook_input):
            vlad_codebook, codebook_metadata = load_vlad_codebook_npz(Path(args.vlad_codebook_input))
            if str(codebook_metadata.get("layer_name", args.layer_name)) != str(args.layer_name):
                raise ValueError("VLAD codebook layer_name does not match --layer_name")
            metadata["vlad_codebook_input"] = str(args.vlad_codebook_input)
        elif args.vlad_codebook_manifest:
            codebook_manifests = [TokenBankManifest.from_json(Path(path)) for path in args.vlad_codebook_manifest]
            vlad_codebook = fit_vlad_codebook_from_manifests(
                codebook_manifests,
                args.layer_name,
                clusters=int(args.vlad_clusters),
                iterations=int(args.vlad_iterations),
                max_tokens=int(args.vlad_max_tokens),
                max_tokens_per_image=int(args.vlad_codebook_tokens_per_image),
                max_images=int(args.vlad_codebook_max_images),
                normalize_tokens=bool(args.normalize_tokens),
                seed=int(args.seed),
            )
            metadata["vlad_codebook_manifests"] = [str(path) for path in args.vlad_codebook_manifest]
        else:
            vlad_codebook = fit_vlad_codebook_from_manifest(
                manifest,
                args.layer_name,
                clusters=int(args.vlad_clusters),
                iterations=int(args.vlad_iterations),
                max_tokens=int(args.vlad_max_tokens),
                max_tokens_per_image=int(args.vlad_codebook_tokens_per_image),
                max_images=int(args.vlad_codebook_max_images),
                normalize_tokens=bool(args.normalize_tokens),
                seed=int(args.seed),
            )
        if str(args.vlad_codebook_output):
            save_vlad_codebook_npz(
                Path(args.vlad_codebook_output),
                vlad_codebook,
                layer_name=str(args.layer_name),
                normalize_tokens=bool(args.normalize_tokens),
                metadata={
                    "token_manifest": str(args.manifest),
                    "token_manifest_sha256": file_sha256_short(Path(args.manifest)),
                    "vlad_clusters": int(vlad_codebook.shape[0]),
                    "vlad_iterations": int(args.vlad_iterations),
                    "vlad_max_tokens": int(args.vlad_max_tokens),
                    "vlad_codebook_tokens_per_image": int(args.vlad_codebook_tokens_per_image),
                    "vlad_codebook_max_images": int(args.vlad_codebook_max_images),
                    "seed": int(args.seed),
                },
            )
            metadata["vlad_codebook_output"] = str(args.vlad_codebook_output)
    bank = build_token_descriptor_bank(
        manifest=manifest,
        layer_name=args.layer_name,
        pooling=args.pooling,
        gem_power=args.gem_power,
        normalize_tokens=bool(args.normalize_tokens),
        normalize=not args.no_normalize,
        metadata=metadata,
        vlad_clusters=int(args.vlad_clusters),
        vlad_iterations=int(args.vlad_iterations),
        vlad_max_tokens=int(args.vlad_max_tokens),
        seed=int(args.seed),
        vlad_codebook=vlad_codebook,
        vlad_tokens_per_image=int(args.vlad_tokens_per_image),
        descriptor_power=float(args.descriptor_power),
    )
    if args.pca_whitening_input or args.pca_whitening_output:
        pca_metadata = dict(bank.metadata or {})
        if args.pca_whitening_input:
            transform = load_pca_whitening_transform_npz(Path(args.pca_whitening_input))
            pca_metadata["pca_whitening_input"] = str(args.pca_whitening_input)
        else:
            if int(args.pca_dim) <= 0:
                raise ValueError("--pca_dim must be positive when fitting PCA whitening")
            transform = fit_pca_whitening_transform(
                bank.descriptors,
                output_dim=int(args.pca_dim),
                whitening_epsilon=float(args.pca_whitening_epsilon),
            )
            save_pca_whitening_transform_npz(Path(args.pca_whitening_output), transform)
            pca_metadata["pca_whitening_output"] = str(args.pca_whitening_output)
        projected = apply_pca_whitening_transform(bank.descriptors, transform, normalize=not args.no_normalize)
        pca_metadata["pca_dim"] = int(projected.shape[1])
        bank = TokenDescriptorBank(
            image_ids=bank.image_ids,
            descriptors=projected,
            layer_name=bank.layer_name,
            pooling=f"{bank.pooling}+pca_whiten",
            normalized=not args.no_normalize,
            metadata=pca_metadata,
        )
    bank.to_npz(Path(args.output))


if __name__ == "__main__":
    main()
