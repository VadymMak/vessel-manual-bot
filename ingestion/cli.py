"""CLI индексации: python3 -m ingestion.cli manual.pdf [--out chunks.json]"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .pipeline import ingest, parse_title_page, to_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Индексация PDF технической документации")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("chunks.json"))
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=None)
    args = ap.parse_args()

    # Профиль печатается ДО остальной сводки: если он определился неверно,
    # все числа ниже недостоверны, и человек должен увидеть это первым.
    chunks = ingest(args.pdf, args.first, args.last,
                    on_profile=lambda p: print(p.describe()))
    document = parse_title_page(args.pdf)
    to_json(chunks, args.out, document)

    lengths = sorted(len(c.content) for c in chunks)
    print(f"Документ:  {args.pdf.name}")
    print(f"Модели:    {', '.join(document['applicable_models']) or '— (нет на титуле)'}")
    print(f"Серийные:  {len(document['serial_prefixes'])} префиксов"
          f"{' — ' + ', '.join(document['serial_prefixes'][:5]) + ', …' if document['serial_prefixes'] else ''}")
    print(f"Чанков:    {len(chunks)}")
    print(f"Типы:      {dict(Counter(c.chunk_type for c in chunks))}")
    print(f"С WARNING: {sum(1 for c in chunks if c.has_warning)}")
    if lengths:
        print(
            f"Длина:     медиана={lengths[len(lengths) // 2]} "
            f"p90={lengths[int(len(lengths) * 0.9)]} max={lengths[-1]}"
        )
    print(f"Записано:  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
