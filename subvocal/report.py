"""M5: one HTML report per prompt, combining a :class:`~subvocal.profile.Profile`
summary (plain tables -- no charting) with ``jlens.vis``'s own interactive d3
slice view, embedded unmodified in an ``<iframe>``. Per CLAUDE.md: "reusing
the d3 slice view from ../jacobian-lens. Do not build a new visualization."
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import jlens.vis

from subvocal.lens import FittedLens
from subvocal.metrics import build_profile, reindex_to_depth, subsample_layers
from subvocal.profile import Profile

PageMode = Literal["embed", "fetch"]


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _boundaries_section(profile: Profile) -> str:
    b = profile.boundaries()
    rows = [
        ("top-k accuracy", f"{b.topk_accuracy:.1f}"),
        ("kurtosis", f"{b.kurtosis:.1f}"),
        ("autocorrelation", f"{b.autocorrelation:.1f}"),
        ("effective dim", f"{b.effective_dim:.1f}"),
        ("CKA", f"{b.cka:.1f}"),
    ]
    flag = (
        '<p class="flag">⚠ disagreement: signals span more than 10% of depth</p>'
        if b.disagreement
        else '<p class="ok">signals agree within 10% of depth</p>'
    )
    return (
        "<h2>Boundary signals (depth 0-100)</h2>"
        + _table(("signal", "depth"), rows)
        + flag
    )


def _fve_occupancy_section(profile: Profile, n_layers: int) -> str:
    fve = profile.fve()
    occ = profile.occupancy()
    rows = [
        (
            layer,
            f"{reindex_to_depth(layer, n_layers):.0f}",
            f"{fve[li]:.4f}",
            f"{occ[:, li].mean():.2f}",
        )
        for li, layer in enumerate(profile.layers)
    ]
    return (
        "<h2>FVE / occupancy per sampled layer</h2>"
        + _table(("layer", "depth", "fve", "mean occupancy"), rows)
    )


def _loadings_section(profile: Profile, concepts: Sequence[str]) -> str:
    if not concepts:
        return ""
    rows = []
    for concept in concepts:
        grid = profile.loading(concept)
        trace = grid.mean(axis=0)
        peak_idx = int(trace.argmax())
        rows.append(
            (
                concept,
                profile.layers[peak_idx],
                f"{trace[peak_idx]:.3f}",
            )
        )
    return (
        "<h2>Concept loading peaks</h2>"
        + _table(("concept", "peak layer", "peak loading"), rows)
    )


_PAGE_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { font-size: 1.3rem; }
h2 { font-size: 1.05rem; margin-top: 1.5rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; font-size: 0.9rem; }
th { background: #f2f2f2; }
p.flag { color: #b23; font-weight: 600; }
p.ok { color: #2a6; }
.prompt { font-family: ui-monospace, monospace; background: #f7f7f7; padding: 0.6rem; border-radius: 4px; }
iframe { width: 100%; height: var(--iframe-height, 620px); border: 1px solid #ccc; margin-top: 1rem; }
"""


def build_report(
    lens: FittedLens,
    prompt: str,
    *,
    profile: Profile | None = None,
    concepts: Sequence[str] = (),
    title: str | None = None,
    layers: Sequence[int] | None = None,
    iframe_height: int = 620,
    mode: PageMode = "embed",
    out_dir: str | Path | None = None,
    slice_kwargs: dict | None = None,
) -> str:
    """Render one prompt's :class:`Profile` summary plus ``jlens``'s own
    interactive slice view into a single self-contained HTML page.

    Args:
        lens: The fitted lens to profile and visualize with.
        prompt: The prompt to report on.
        profile: A precomputed :class:`~subvocal.profile.Profile`. Computed
            via :func:`~subvocal.metrics.build_profile` if not given.
        concepts: Concepts to show peak-loading rows for (also passed to
            :func:`~subvocal.metrics.build_profile` when ``profile`` isn't
            given).
        title: Page title. Defaults to a truncated ``prompt``.
        layers: Layers for the ``Profile`` computation, when ``profile``
            isn't given. Defaults to :func:`~subvocal.metrics.subsample_layers`.
        iframe_height: Height (px) of the embedded slice-view iframe.
        mode: ``jlens.vis.build_page``'s page mode -- ``"embed"`` (default)
            produces one fully self-contained file; ``"fetch"`` writes
            sidecar files to ``out_dir`` and needs no network access to
            build (no d3 CDN fetch at build time), so it's what this
            package's own tests use.
        out_dir: Required when ``mode="fetch"``; where the slice view's
            sidecar files are written (the summary/iframe page itself is
            still returned, not written -- write it into the same directory
            if you want one bundle).
        slice_kwargs: Extra keyword arguments forwarded to
            ``jlens.vis.compute_slice`` (e.g. ``top_n``, ``mask_display``).

    Returns:
        The rendered page as an HTML string.
    """
    if profile is None:
        layers = list(layers) if layers is not None else subsample_layers(lens.n_layers)
        profile = build_profile(lens, prompt, concepts=list(concepts), layers=layers)

    title = title or f"subvocal report: {prompt[:60]!r}"
    slice_data = jlens.vis.compute_slice(
        lens.model, lens.jacobian_lens, prompt, **(slice_kwargs or {})
    )
    slice_page, _, _ = jlens.vis.build_page(
        slice_data,
        prompt,
        title=f"{title} (slice view)",
        description="jlens's own d3 slice view, reused unmodified.",
        mode=mode,
        out_dir=out_dir,
    )

    body = (
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="prompt">{html.escape(prompt)}</p>'
        + _boundaries_section(profile)
        + _fve_occupancy_section(profile, lens.n_layers)
        + _loadings_section(profile, list(concepts))
        + "<h2>Slice view (jlens.vis, unmodified)</h2>"
        + f'<iframe style="--iframe-height:{iframe_height}px" '
        + f'srcdoc="{html.escape(slice_page)}"></iframe>'
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_PAGE_CSS}</style>"
        f"</head><body>{body}</body></html>"
    )
