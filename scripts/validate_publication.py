#!/usr/bin/env python3
"""Прототип входного шлюза для админ-панели VesselManualBot.
 
Пять проверок перед индексацией. Любая FAIL — документ не принимается.
Использование:  python3 validate_publication.py <файл.pdf>
"""
import collections
import re
import sys
 
import fitz
 
# Серийные префиксы двигателей судна. Берутся из титульных страниц
# уже проиндексированных O&M, а не задаются руками.
VESSEL_PREFIXES = {
    "S2A", "S2B", "S2D", "S2E", "S2F", "S2G", "S2H", "S2J", "S2K", "S2L",
    "S2M", "S2N", "S2P", "S2R", "S2S", "S2T", "S2W", "S2X", "S2Y", "S2Z",
    "B5G", "MRG", "PAG", "MKH",          # 3500B/3500C Marine
    "T2P",                                # C18 Marine
}
 
PUB = re.compile(r"\b(?:SEBU|SENR|RENR|UENR|KENR|SEBF|REHS|SEHS|PEHP)\d{4}(?:-\d{2})?\b")
ICODE = re.compile(r"i\d{8}")
PREFIX = re.compile(r"\b([A-Z0-9]{3})\d?1?\s?-\s?[Uu][Pp]\b")
 
MIN_ICODE_DENSITY = 0.35
MIN_FOOTER_RATIO = 0.80
MIN_PAGES = 20
 
 
def validate(path):
    doc = fitz.open(path)
    pages = [doc[i].get_text() for i in range(doc.page_count)]
    full = "\n".join(pages)
    n = len(pages)
 
    icodes = set(ICODE.findall(full))
    density = len(icodes) / n if n else 0
    footer = sum(1 for t in pages if PUB.search(t)) / n if n else 0
    codes = collections.Counter(PUB.findall(full))
    prefixes = set(PREFIX.findall(full))
    overlap = prefixes & VESSEL_PREFIXES
 
    checks = [
        ("объём",              n >= MIN_PAGES,
         f"{n} стр. (минимум {MIN_PAGES})"),
        ("плотность i-кодов",  density >= MIN_ICODE_DENSITY,
         f"{density:.2f}/стр, всего {len(icodes)} (минимум {MIN_ICODE_DENSITY})"),
        ("код публикации",     footer >= MIN_FOOTER_RATIO,
         f"{footer*100:.0f}% страниц (минимум {MIN_FOOTER_RATIO*100:.0f}%)"),
        ("публикация опознана", bool(codes),
         codes.most_common(1)[0][0] if codes else "не найдено"),
        # Документ без серийных префиксов вообще (SENR3130 «All Caterpillar
        # Products») применим ко всему и проходит. Отклоняется только тот,
        # у кого префиксы ЕСТЬ и они чужие.
        ("серийные префиксы",  bool(overlap) or not prefixes,
         f"совпало {sorted(overlap)}" if overlap
         else "префиксов нет — документ универсальный" if not prefixes
         else f"НЕТ пересечения; в документе {sorted(prefixes)[:12]}"),
    ]
 
    print(f"\n{'='*70}\n{path.split('/')[-1]}\n{'='*70}")
    print(f"  {pages[0].strip().splitlines()[0][:60] if pages[0].strip() else '(нет текста)'}")
    print()
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'OK  ' if passed else 'FAIL'}]  {name:<20} {detail}")
    print(f"\n  ВЕРДИКТ: {'ПРИНЯТЬ К ИНДЕКСАЦИИ' if ok else 'НЕ ИНДЕКСИРОВАТЬ'}\n")
    return ok
 
 
if __name__ == "__main__":
    for p in sys.argv[1:]:
        validate(p)