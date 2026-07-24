"""Tests for report.py. Uses mode="fetch" (sidecar files, no CDN fetch at
build time) against the tiny_fitted_lens fixture, so these run without
network access -- mode="embed" is exercised only in the ad hoc README example
script, which does have network access.
"""

from __future__ import annotations

from subvocal import report

_PROMPT = "The quick brown fox jumps over the lazy dog near the river bank today."


class TestBuildReport:
    def test_returns_self_contained_html_string(self, tiny_fitted_lens, tmp_path):
        page = report.build_report(
            tiny_fitted_lens,
            _PROMPT,
            concepts=[" fox"],
            layers=[0, 1, 2],
            mode="fetch",
            out_dir=tmp_path,
        )
        assert page.startswith("<!doctype html>")
        assert "<iframe" in page
        assert "srcdoc=" in page

    def test_includes_prompt_and_boundary_signals(self, tiny_fitted_lens, tmp_path):
        page = report.build_report(
            tiny_fitted_lens,
            _PROMPT,
            layers=[0, 1, 2],
            mode="fetch",
            out_dir=tmp_path,
        )
        assert _PROMPT in page
        assert "top-k accuracy" in page
        assert "kurtosis" in page

    def test_includes_concept_loading_peaks_when_given(self, tiny_fitted_lens, tmp_path):
        page = report.build_report(
            tiny_fitted_lens,
            _PROMPT,
            concepts=[" fox", " dog"],
            layers=[0, 1, 2],
            mode="fetch",
            out_dir=tmp_path,
        )
        assert "Concept loading peaks" in page
        assert " fox" in page
        assert " dog" in page

    def test_omits_loadings_section_when_no_concepts(self, tiny_fitted_lens, tmp_path):
        page = report.build_report(
            tiny_fitted_lens,
            _PROMPT,
            layers=[0, 1, 2],
            mode="fetch",
            out_dir=tmp_path,
        )
        assert "Concept loading peaks" not in page

    def test_accepts_precomputed_profile(self, tiny_fitted_lens, tmp_path):
        from subvocal.metrics import build_profile

        profile = build_profile(
            tiny_fitted_lens, _PROMPT, concepts=[" fox"], layers=[0, 1, 2]
        )
        page = report.build_report(
            tiny_fitted_lens,
            _PROMPT,
            profile=profile,
            concepts=[" fox"],
            mode="fetch",
            out_dir=tmp_path,
        )
        assert "<!doctype html>" in page

    def test_writes_sidecar_files_in_fetch_mode(self, tiny_fitted_lens, tmp_path):
        report.build_report(
            tiny_fitted_lens,
            _PROMPT,
            layers=[0, 1, 2],
            mode="fetch",
            out_dir=tmp_path,
        )
        assert (tmp_path / "meta.json").exists()
        assert (tmp_path / "slice.bin").exists()
