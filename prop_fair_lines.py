#!/usr/bin/env python3
"""
Prop fair-line finder for NFL + college football (for use alongside PrizePicks).

Pulls player-prop odds from many US sportsbooks (The Odds API), removes the sportsbook margin,
and publishes a "fair line" per player/stat plus the PrizePicks line thresholds where
Over / Under becomes a good pick. You then compare to the line shown in the PrizePicks app.

  python prop_fair_lines.py --sports=auto|NFL|NCAAF|both  [--hours=30]

Env: ODDS_API_KEY, NTFY_TOPIC (GITHUB_REPOSITORY is set automatically on GitHub)
Not financial/betting advice. Probabilities are estimates.
"""
import csv, math, os, statistics, sys
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import requests

API = "https://api.the-odds-api.com/v4"
KEY = os.getenv("ODDS_API_KEY", "")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "change-me")
REPO = os.getenv("GITHUB_REPOSITORY", "")
ROOT = Path(__file__).parent
OUT, HIST = ROOT / "output", ROOT / "history"

# ---------------- settings you can tweak ----------------
TARGET_HIT = 0.58     # a pick should hit at least this often to be worth playing (2-4 pick entries)
MIN_BOOKS = 3         # need at least this many sportsbooks posting the prop
MAX_EVENTS = {"NFL": 16, "NCAAF": 8}   # games per run (each game costs API credits)
SPORTS = {"NFL": "americanfootball_nfl", "NCAAF": "americanfootball_ncaaf"}
# stat key -> (label, rough std-dev of the stat given its average). Starter values: refine with results.
STATS = {
    "player_pass_yds": ("PASS YDS", lambda m: 0.27 * m + 8),
    "player_rush_yds": ("RUSH YDS", lambda m: 0.50 * m + 5),
    "player_reception_yds": ("REC YDS", lambda m: 0.60 * m + 6),
    "player_receptions": ("RECEPTIONS", lambda m: 1.05 * sqrt(max(m, 0.1))),
    "player_pass_tds": ("PASS TDS", lambda m: 0.85 * sqrt(max(m, 0.1)) + 0.1),
}
ND = NormalDist()
requests_left = None


# ---------------- API ----------------
def get(path, **params):
    global requests_left
    r = requests.get(API + path, params={"apiKey": KEY, "dateFormat": "iso", **params}, timeout=40)
    requests_left = r.headers.get("x-requests-remaining", requests_left)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} {r.text[:150]}")
    return r.json()


def pick_events(sport_key, label, hours):
    """1-credit call: games starting soon, ranked by how many books price them (marquee games first)."""
    now = datetime.now(timezone.utc)
    evs = get(f"/sports/{sport_key}/odds", regions="us", markets="totals", oddsFormat="decimal")
    soon = [e for e in evs if now - timedelta(hours=1) <= ts(e["commence_time"]) <= now + timedelta(hours=hours)]
    soon.sort(key=lambda e: len(e.get("bookmakers", [])), reverse=True)
    return soon[:MAX_EVENTS[label]]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def event_props(sport_key, event_id):
    markets = ",".join(STATS)
    try:
        return get(f"/sports/{sport_key}/events/{event_id}/odds", regions="us", markets=markets, oddsFormat="decimal")
    except RuntimeError as e:
        if "422" not in str(e):
            raise
        # some market isn't offered for this sport: try them one at a time
        merged = None
        for m in STATS:
            try:
                d = get(f"/sports/{sport_key}/events/{event_id}/odds", regions="us", markets=m, oddsFormat="decimal")
            except RuntimeError:
                continue
            if merged is None:
                merged = d
            else:
                merged["bookmakers"] += d.get("bookmakers", [])
        return merged or {"bookmakers": []}


# ---------------- math ----------------
def devig_over(over_price, under_price):
    io, iu = 1 / over_price, 1 / under_price
    return io / (io + iu)


def implied_mean(point, p_over, sigma_fn):
    """Mean of the stat implied by one book's line + no-vig Over probability (normal approximation)."""
    z = ND.inv_cdf(min(max(p_over, 0.02), 0.98))
    mu = point
    for _ in range(4):
        mu = point + sigma_fn(max(mu, 0.1)) * z
    return mu


def down_half(x): return math.floor(x * 2) / 2
def up_half(x): return math.ceil(x * 2) / 2


def analyze(event, sport_label):
    game = f"{event.get('away_team', '?')} @ {event.get('home_team', '?')}"
    seen = {}   # (market, player) -> list of (point, p_over)
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] not in STATS:
                continue
            by = {}
            for o in mk.get("outcomes", []):
                if o.get("point") is None or not o.get("description"):
                    continue
                by.setdefault((o["description"], o["point"]), {})[o["name"]] = o["price"]
            for (player, point), sides in by.items():
                if "Over" in sides and "Under" in sides and sides["Over"] > 1 and sides["Under"] > 1:
                    seen.setdefault((mk["key"], player), []).append((point, devig_over(sides["Over"], sides["Under"])))
    rows = []
    z_t = ND.inv_cdf(TARGET_HIT)
    for (mkey, player), obs in seen.items():
        label, sigma_fn = STATS[mkey]
        if len(obs) < MIN_BOOKS:
            continue
        mus = [implied_mean(pt, p, sigma_fn) for pt, p in obs]
        fair = statistics.median(mus)
        sigma = sigma_fn(fair)
        spread = statistics.pstdev(mus) if len(mus) > 1 else 0
        conf = "HIGH" if len(obs) >= 5 and spread <= 0.12 * sigma else \
               "MED" if spread <= 0.25 * sigma else "LOW"
        rows.append({"sport": sport_label, "game": game, "start": event.get("commence_time", ""),
                     "player": player, "stat": label, "fair": round(fair, 1),
                     "books": len(obs), "book_line": statistics.median([pt for pt, _ in obs]),
                     "over_if_pp_at_or_below": down_half(fair - sigma * z_t),
                     "under_if_pp_at_or_above": up_half(fair + sigma * z_t),
                     "agreement": conf, "spread": round(spread, 2)})
    return rows


# ---------------- output ----------------
def write_outputs(rows, label_tag):
    OUT.mkdir(exist_ok=True); HIST.mkdir(exist_ok=True)
    cols = ["sport", "game", "start", "player", "stat", "fair", "book_line", "books",
            "over_if_pp_at_or_below", "under_if_pp_at_or_above", "agreement", "spread"]
    for path in (OUT / "latest.csv", HIST / f"{datetime.now(timezone.utc):%Y-%m-%d}_{label_tag}.csv"):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            w.writerows([{c: r[c] for c in cols} for r in rows])
    md = [f"# Fair lines - updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n",
          f"Over is a ~{TARGET_HIT:.0%}+ pick if the PrizePicks line is **at or below** the Over number. "
          "Under likewise if **at or above** the Under number. Otherwise skip.\n"]
    game = None
    for r in sorted(rows, key=lambda r: (r["start"], r["game"], r["player"], r["stat"])):
        if r["game"] != game:
            game = r["game"]
            md.append(f"\n## {game}\n\n| Player | Stat | Fair | Over if PP <= | Under if PP >= | Books | Agreement |\n|---|---|---|---|---|---|---|")
        md.append(f"| {r['player']} | {r['stat']} | {r['fair']} | {r['over_if_pp_at_or_below']} | "
                  f"{r['under_if_pp_at_or_above']} | {r['books']} | {r['agreement']} |")
    (OUT / "latest.md").write_text("\n".join(md) + "\n")


def notify(title, body, url=None, prio="default"):
    print(f"NOTIFY: {title}\n{body}\n")
    h = {"Title": title.encode("ascii", "replace").decode(), "Priority": prio}
    if url:
        h["Click"] = url
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=body.encode("utf-8"), headers=h, timeout=20)
    except Exception as e:
        print("ntfy failed:", e)


def main():
    arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--sports=")), "auto")
    hours = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--hours=")), "30"))
    wd = datetime.now(timezone.utc).weekday()
    labels = {"NFL": ["NFL"], "NCAAF": ["NCAAF"], "both": ["NFL", "NCAAF"]}.get(
        arg, ["NCAAF"] if wd == 5 else ["NFL"] if wd == 6 else ["NFL", "NCAAF"])
    if not KEY:
        notify("Prop finder: API key missing", "Add the ODDS_API_KEY secret in the repo settings.", prio="high")
        sys.exit(1)
    rows, notes = [], []
    for lab in labels:
        try:
            events = pick_events(SPORTS[lab], lab, hours)
        except Exception as e:
            notes.append(f"{lab}: could not list games ({e})"); continue
        if not events:
            notes.append(f"{lab}: no games in the next {hours}h"); continue
        got = 0
        for ev in events:
            try:
                r = analyze(event_props(SPORTS[lab], ev["id"]) | {"home_team": ev["home_team"],
                            "away_team": ev["away_team"], "commence_time": ev["commence_time"]}, lab)
                rows += r; got += len(r)
            except Exception as e:
                notes.append(f"{lab} {ev.get('away_team')}@{ev.get('home_team')}: {str(e)[:60]}")
        notes.append(f"{lab}: {len(events)} games checked, {got} props with enough books")
    link = f"https://github.com/{REPO}/blob/main/output/latest.md" if REPO else None
    if not rows:
        notify("Prop finder: nothing found", "\n".join(notes) + f"\nAPI credits left: {requests_left}", link)
        return
    write_outputs(rows, "_".join(labels))
    good = [r for r in rows if r["agreement"] in ("HIGH", "MED")]
    good.sort(key=lambda r: (r["agreement"] != "HIGH", -r["books"], r["start"]))
    lines = [f"{r['player']} {r['stat']}: fair {r['fair']} | OVER if PP <= {r['over_if_pp_at_or_below']} "
             f"| UNDER if PP >= {r['under_if_pp_at_or_above']} [{r['agreement']}, {r['books']} books]"
             for r in good[:22]]
    body = (f"HOW TO USE: find the player in PrizePicks. If its line is at/below the OVER number, Over is a "
            f"{TARGET_HIT:.0%}+ pick; at/above the UNDER number, Under is. Otherwise skip. Build 2-4 pick "
            "entries from picks in DIFFERENT games.\n\n" + "\n".join(lines)
            + f"\n\n{len(rows)} props total (tap for full list). API credits left: {requests_left}\n"
            + "\n".join(notes))
    notify(f"Fair lines ready: {len(rows)} props", body[:3900], link, "default")


if __name__ == "__main__":
    main()
