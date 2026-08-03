#!/usr/bin/env python3
"""
Офлайн-перебор вариантов слияния. Ни базы, ни эмбеддера, ни реранкера —
только два списка рангов из eval_ranks.json. Все 38 вопросов, все варианты,
оба плеча считаются меньше чем за секунду.

  python3 -m scripts.dump_ranks     # один раз, ~3 минуты
  python3 -m scripts.fuse_lab       # сколько угодно раз, мгновенно

ЧТО МЕРИТСЯ. Ранг целевого чанка в СЛИТОМ списке. Реранкер сюда не входит
сознательно: он видит только то, что слияние ему передало, и вопрос задачи
именно в том, что до него доходит. Ранг в слитом списке — это и есть
то, чем распоряжается RRF_TOP_K.

ГРУППЫ. Разделение не косметическое: 26 якорных вопросов — то, что уже
работает и что нельзя сломать; 10 ru_no_anchor — то, что чиним.
Средним по всем 38 эти две группы мерить нельзя, они разной природы.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import click

# Вопросы без латинского якоря среди старых 28. Замер 2026-08-03: якорь есть
# у 26 из 28, нет ровно у этих двух, и прочерк в sparse стоит ровно у них.
_OLD_NO_ANCHOR = {"gs022", "gs031"}


def _ranks(lst: list[int]) -> dict[int, int]:
    return {cid: i + 1 for i, cid in enumerate(lst)}


def _ranks_nonzero(lst: list[int], scores: list[float], eps=1e-9) -> dict[int, int]:
    """
    Ранги только тех, у кого скор ОТЛИЧЕН ОТ НУЛЯ.

    На кириллическом запросе sparse даёт ровно ноль почти всему окну: замер
    2026-08-03 — у 8 из 10 вопросов ru_no_anchor ненулевой скор имеют 0 или 1
    строка из 50. Порядок среди нулей Postgres выбирает произвольно, и он
    не совпадает между сессиями (пересечение окна gs035 между двумя выгрузками
    составило 0 из 50). Эти строки не свидетельства, а шум, и слияние обязано
    их не видеть.
    """
    out, rank = {}, 0
    for cid, sc in zip(lst, scores):
        if abs(sc) <= eps:
            continue
        rank += 1
        out[cid] = rank
    return out


# ─── варианты слияния ────────────────────────────────────────────────────────
# Каждый принимает (ранг в dense | None, ранг в sparse | None) и возвращает
# скор. None означает «в окне нет».

def v_control(k=60):
    def f(rd, rs):
        s = 0.0
        if rd: s += 1.0 / (k + rd)
        if rs: s += 1.0 / (k + rs)
        return s
    return f


def v_impute(k=60, miss=51):
    """Отсутствие в окне — не ноль слагаемых, а один штрафной ранг."""
    def f(rd, rs):
        return 1.0 / (k + (rd or miss)) + 1.0 / (k + (rs or miss))
    return f


def v_max(k=60):
    """Максимум вместо суммы: присутствие во второй ветви не даёт бонуса."""
    def f(rd, rs):
        return max(1.0 / (k + rd) if rd else 0.0,
                   1.0 / (k + rs) if rs else 0.0)
    return f


def v_weighted(ws, k=60, wd=1.0):
    def f(rd, rs):
        s = 0.0
        if rd: s += wd / (k + rd)
        if rs: s += ws / (k + rs)
        return s
    return f


# (имя, функция скора, отбрасывать ли sparse-строки с нулевым скором)
VARIANTS: list[tuple[str, object, bool]] = [
    ("контроль RRF k=60",          v_control(60),      False),
    ("ОТБРОСИТЬ нулевой sparse",   v_control(60),      True),
    ("вменение 51, k=60",          v_impute(60, 51),   False),
    ("максимум, k=60",             v_max(60),          False),
    ("вес sparse 1.0 (=контроль)", v_weighted(1.0),    False),
    ("вес sparse 0.7",             v_weighted(0.7),    False),
    ("вес sparse 0.5",             v_weighted(0.5),    False),
    ("вес sparse 0.3",             v_weighted(0.3),    False),
    ("вес sparse 0.1",             v_weighted(0.1),    False),
    ("вес sparse 0 (только dense)", v_weighted(0.0),   False),
    ("контроль k=30",              v_control(30),      False),
    ("контроль k=20",              v_control(20),      False),
    ("контроль k=10",              v_control(10),      False),
    ("вменение 51, k=30",          v_impute(30, 51),   False),
    ("вменение 51, k=20",          v_impute(20, 51),   False),
    ("вменение 51, k=10",          v_impute(10, 51),   False),
    ("отбросить нули + вменение",  v_impute(60, 51),   True),
    ("отбросить нули + k=20",      v_control(20),      True),
]


def fuse_rank(arm: dict, targets: list[int], fn, drop_zero=False) -> int | None:
    """Ранг лучшей цели в слитом списке. None — цели нет ни в одном окне."""
    dr = _ranks(arm["dense"])
    sr = (_ranks_nonzero(arm["sparse"], arm["sparse_scores"])
          if drop_zero and "sparse_scores" in arm else _ranks(arm["sparse"]))
    ids = set(dr) | set(sr)
    if not ids:
        return None
    scored = sorted(ids, key=lambda c: (-fn(dr.get(c), sr.get(c)), c))
    hits = [i + 1 for i, c in enumerate(scored) if c in targets]
    return min(hits) if hits else None


def _mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def _median(ranks: list[int | None], miss=999) -> float:
    return statistics.median([r if r else miss for r in ranks]) if ranks else 0.0


def evaluate(data: dict, arm_name: str, fn, drop_zero=False) -> dict:
    anchored, ru, skipped = {}, {}, []
    for qid, rec in data["questions"].items():
        arm = rec["arms"][arm_name]
        if not arm.get("target_reachable", True):
            skipped.append(qid)          # цель отсечена фильтром — не провал
            continue
        r = fuse_rank(arm, rec["targets"], fn, drop_zero)
        (ru if rec["category"] == "ru_no_anchor" else anchored)[qid] = r
    anchored = {q: r for q, r in anchored.items() if q not in _OLD_NO_ANCHOR}
    return {"anchored": anchored, "ru": ru, "skipped": skipped}


@click.command()
@click.option("--ranks", default="eval_ranks.json", show_default=True)
@click.option("--arm", default="nofilter,m3512B", show_default=True)
def main(ranks, arm):
    data = json.loads(Path(ranks).read_text())
    top_k = data["meta"]["rrf_top_k"]

    for arm_name in arm.split(","):
        base = evaluate(data, arm_name, VARIANTS[0][1], VARIANTS[0][2])
        base_anchor = base["anchored"]
        click.echo(f"\n{'='*104}")
        click.echo(f"ПЛЕЧО {arm_name}   RRF_TOP_K={top_k}   "
                   f"якорных {len(base_anchor)}, ru_no_anchor {len(base['ru'])}"
                   + (f", отсечено фильтром {len(base['skipped'])}: "
                      f"{', '.join(base['skipped'])}" if base["skipped"] else ""))
        click.echo("=" * 104)
        click.echo(f"{'вариант':<28}{'26 якорных':^28}{'10 ru_no_anchor':^30}{'все':^16}")
        click.echo(f"{'':<28}{'хуже':>6}{'макс':>7}{'медиана':>9}"
                   f"{'медиана':>10}{'топ-20':>8}{'топ-6':>7}{'MRR':>8}"
                   f"{'медиана':>9}{'MRR':>8}")
        click.echo("-" * 104)

        for name, fn, dz in VARIANTS:
            res = evaluate(data, arm_name, fn, dz)
            a, r = res["anchored"], res["ru"]
            worse = [(q, base_anchor[q], a[q]) for q in a
                     if (a[q] or 999) > (base_anchor[q] or 999)]
            max_worse = max((n - o for _, o, n in
                             [(q, ob or 999, nb or 999) for q, ob, nb in worse]),
                            default=0)
            allr = list(a.values()) + list(r.values())
            ok = not worse
            click.secho(
                f"{name:<28}{len(worse):>6}{max_worse:>7}{_median(list(a.values())):>9.1f}"
                f"{_median(list(r.values())):>10.1f}"
                f"{sum(1 for x in r.values() if x and x <= top_k):>8}"
                f"{sum(1 for x in r.values() if x and x <= 6):>7}"
                f"{_mrr(list(r.values())):>8.3f}"
                f"{_median(allr):>9.1f}{_mrr(allr):>8.3f}"
                + ("" if ok else f"   ОТВЕРГНУТ: {', '.join(q for q,_,_ in worse[:4])}"),
                fg=None if ok else "red")

    # ─── gs033: при каком RRF_TOP_K цель возвращается в окно ─────────────────
    click.echo(f"\n{'='*104}\ngs033 под M=\"3512B\" — ранг цели в слитом списке "
               f"(сейчас RRF_TOP_K={top_k}, цель до реранкера не доходит)")
    click.echo("=" * 104)
    rec = data["questions"]["gs033"]
    for name, fn, dz in VARIANTS:
        r = fuse_rank(rec["arms"]["m3512B"], rec["targets"], fn, dz)
        verdict = ("в окне" if r and r <= top_k
                   else f"нужен RRF_TOP_K ≥ {r}" if r else "цели нет в окнах")
        click.secho(f"  {name:<30}ранг {str(r):>5}   {verdict}",
                    fg="green" if r and r <= top_k else None)
    click.echo()


if __name__ == "__main__":
    sys.exit(main())
