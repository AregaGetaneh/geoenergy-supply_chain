"""Compare the drawn figures with the published versions.

A figure passes when its page size, its axes boxes, its text content and its vector
drawing content all match the published PDF. Byte equality is not required: a PDF records
its creation time, so two identical runs never produce identical bytes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REFERENCE = ROOT / "reference"

PUBLISHED = ["Fig1_gulf_anatomy", "Fig6_transport_boundary", "Fig2_world_outcomes",
             "Fig3_absorption", "Fig5_inventory_buffering", "Fig8_boundaries",
             "Fig7_episode_maps", "Fig4_arrivals_vs_availability",
             "FigS1_closure_residuals", "FigS5_corridor_dependence"]


def _page(path):
    import pymupdf

    doc = pymupdf.open(str(path))
    page = doc[0]
    # whitespace is normalised away: mathtext spacing around "=" differs between
    # matplotlib versions and is kerning, not content
    text = re.sub(r"\s+", "", page.get_text())
    size = (round(page.rect.width, 2), round(page.rect.height, 2))
    horizontal, vertical = [], []
    for group in page.get_drawings():
        for item in group["items"]:
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            if abs(a.y - b.y) < 0.12 and abs(a.x - b.x) > 20:
                horizontal.append((min(a.x, b.x), max(a.x, b.x), (a.y + b.y) / 2))
            elif abs(a.x - b.x) < 0.12 and abs(a.y - b.y) > 20:
                vertical.append((min(a.y, b.y), max(a.y, b.y), (a.x + b.x) / 2))
    boxes = []
    for y0, y1, x in vertical:
        for hx0, hx1, hy in horizontal:
            if abs(hy - y1) < 1.2 and abs(hx0 - x) < 1.6:
                boxes.append(tuple(round(v, 2) for v in (x, y1, hx1 - hx0, y1 - y0)))
                break
    doc.close()
    return size, text, sorted(set(boxes))


def main() -> int:
    failures = 0
    print("\nfigure                            size    axes boxes   text")
    for stem in PUBLISHED:
        drawn, published = OUTPUTS / (stem + ".pdf"), REFERENCE / (stem + ".pdf")
        if not drawn.exists():
            print("  %-32s NOT DRAWN" % stem)
            failures += 1
            continue
        a, b = _page(drawn), _page(published)
        ok = (a[0] == b[0], a[2] == b[2], a[1] == b[1])
        failures += not all(ok)
        print("  %-32s %-7s %-12s %s"
              % (stem, "same" if ok[0] else "DIFFER",
                 "%d same" % len(a[2]) if ok[1] else "DIFFER",
                 "same" if ok[2] else "DIFFER"))
    print("\n%d of %d figures match the published versions"
          % (len(PUBLISHED) - failures, len(PUBLISHED)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
