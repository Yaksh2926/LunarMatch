"""Smoke and unit tests for self-contained HTML report generation and quick-look plots."""
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from lunarmatch.models.domain import MatchSet, RegistrationMetrics, RunManifest, TransformEstimate
from lunarmatch.visualization.plots import downsample_image
from lunarmatch.visualization.report import generate_html_report


class SimpleHTMLValidator(HTMLParser):
    """Basic HTML parser to verify tags are well-formed and capture element text."""

    def __init__(self):
        super().__init__()
        self.tags = []
        self.external_urls = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs_dict = dict(attrs)
        # Track external script/style HTTP links
        if tag in ("script", "link", "img"):
            src = attrs_dict.get("src") or attrs_dict.get("href")
            if src and (src.startswith(("http://", "https://"))):
                self.external_urls.append(src)


def test_downsample_image_bounds():
    """Test that image downsampling respects max_dim bound."""
    large_img = np.zeros((1600, 2400), dtype=np.uint8)
    ds = downsample_image(large_img, max_dim=800)

    assert ds.shape[0] <= 800
    assert ds.shape[1] <= 800
    assert ds.shape == (533, 800)


def test_generate_html_report_success_smoke(tmp_path: Path):
    """Smoke test generating HTML report for successful registration run."""
    src_img = np.random.default_rng(42).uniform(0, 255, (200, 200)).astype(np.uint8)
    ref_img = np.random.default_rng(43).uniform(0, 255, (200, 200)).astype(np.uint8)
    warped_img = np.random.default_rng(44).uniform(0, 255, (200, 200)).astype(np.uint8)

    matches = MatchSet(
        source_points=np.array([[10.0, 20.0], [50.0, 60.0]], dtype=np.float64),
        reference_points=np.array([[10.5, 20.2], [50.1, 60.3]], dtype=np.float64),
        scores=np.array([0.95, 0.90], dtype=np.float64),
        inliers=np.array([True, True], dtype=bool),
    )

    transform = TransformEstimate.from_matrix("affine", np.eye(3), inlier_count=2, inlier_ratio=1.0, rmse_px=0.2)
    metrics = RegistrationMetrics(
        rmse_px=0.2,
        inlier_count=2,
        total_matches=2,
        inlier_ratio=1.0,
        occupied_grid_fraction=0.5,
        convex_hull_coverage_fraction=0.5,
        runtime_seconds=0.3,
        warnings=[],
    )

    manifest = RunManifest(
        pair_id="source_<test>&pair",
        source_path="/path/to/source.tif",
        reference_path="/path/to/ref.tif",
        config_hash="abc123hash",
        timestamp_utc="2026-08-23T11:00:00Z",
        package_version="0.1.0",
        seed=42,
        metrics=metrics,
        transform=transform,
        quality_gate_passed=True,
    )

    report_path = tmp_path / "report.html"
    html_content = generate_html_report(
        manifest=manifest,
        matches=matches,
        source_image=src_img,
        reference_image=ref_img,
        warped_image=warped_img,
        output_path=report_path,
        stage_timings={"io": 0.05, "matching": 0.25},
    )

    assert report_path.is_file()
    assert "<title>LunarMatch Registration Report" in html_content
    assert "PASSED" in html_content

    # HTML Escaping verification: '<test>&pair' should be escaped as '&lt;test&gt;&amp;pair'
    assert "&lt;test&gt;&amp;pair" in html_content
    assert "<test>&pair" not in html_content

    # Parse HTML and check offline compatibility (no external HTTP script/style URLs)
    validator = SimpleHTMLValidator()
    validator.feed(html_content)
    assert len(validator.external_urls) == 0, f"Found external HTTP URLs: {validator.external_urls}"
    assert "img" in validator.tags
    assert "data:image/png;base64," in html_content


def test_generate_html_report_failure_smoke(tmp_path: Path):
    """Smoke test generating HTML report for ordinary registration failure run."""
    metrics = RegistrationMetrics(
        rmse_px=None,
        inlier_count=0,
        total_matches=0,
        inlier_ratio=0.0,
        occupied_grid_fraction=0.0,
        convex_hull_coverage_fraction=0.0,
        runtime_seconds=0.05,
        warnings=["Zero keypoints detected in source raster"],
    )

    manifest = RunManifest(
        pair_id="blank_src_blank_ref",
        source_path="/path/blank_src.tif",
        reference_path="/path/blank_ref.tif",
        config_hash="hash_blank",
        timestamp_utc="2026-08-23T11:00:00Z",
        package_version="0.1.0",
        seed=42,
        metrics=metrics,
        transform=None,
        quality_gate_passed=False,
    )

    report_path = tmp_path / "failure_report.html"
    html_content = generate_html_report(
        manifest=manifest,
        matches=None,
        output_path=report_path,
    )

    assert report_path.is_file()
    assert "FAILED" in html_content
    assert "Diagnostic Warnings & Failures" in html_content
    assert "Zero keypoints detected in source raster" in html_content

    validator = SimpleHTMLValidator()
    validator.feed(html_content)
    assert len(validator.external_urls) == 0
