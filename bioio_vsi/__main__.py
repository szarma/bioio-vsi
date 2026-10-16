"""CLI entry point: python -m bioio_vsi describe FILE"""

from __future__ import annotations

import argparse
import sys

from bioio_vsi import describe
from bioio_vsi.errors import VsiError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioio_vsi", description="Inspect cellSens VSI files")
    sub = parser.add_subparsers(dest="command", required=True)
    anon = sub.add_parser("anonymize", help="Write an anonymised structure fixture")
    anon.add_argument("path", help="Path to a .vsi file")
    anon.add_argument("out_dir", help="Output directory for anon.vsi and .skel files")
    anon.add_argument(
        "--keep-numbers",
        action="store_true",
        help="Keep measured values (pixel size, z, times, wavelengths, device settings) "
        "as stored instead of replacing them with synthetic ones",
    )
    mat = sub.add_parser("materialize", help="Rebuild a viewable file from a fixture")
    mat.add_argument("fixture_dir", help="Directory containing anon.vsi")
    mat.add_argument("out_dir", help="Output directory")
    mat.add_argument(
        "--planes",
        choices=("default", "all"),
        default="default",
        help=(
            "Which planes receive synthetic pixels (default: the first plane at each "
            "level of at most 64 MiB, and always at the smallest level)"
        ),
    )
    desc = sub.add_parser("describe", help="Print metadata without reading pixels")
    desc.add_argument("path", help="Path to a .vsi file")
    desc.add_argument(
        "--redact",
        action="store_true",
        help="Omit paths and names (safe to attach to public issues)",
    )
    args = parser.parse_args(argv)
    if args.command == "anonymize":
        from bioio_vsi._anonymize import anonymize

        try:
            out = anonymize(args.path, args.out_dir, synthetic_numbers=not args.keep_numbers)
        except VsiError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        except OSError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        sys.stdout.write(f"{out}\n")
        return 0
    if args.command == "materialize":
        from bioio_vsi._materialize import materialize

        try:
            out = materialize(args.fixture_dir, args.out_dir, planes=args.planes)
        except VsiError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        except OSError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        sys.stdout.write(f"{out}\n")
        return 0
    if args.command == "describe":
        if args.redact:
            from bioio_vsi._describe_redact import describe_redacted

            text, has_error = describe_redacted(args.path)
            sys.stdout.write(text)
            return 1 if has_error else 0
        try:
            sys.stdout.write(describe(args.path, redact=False))
        except VsiError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        except OSError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
