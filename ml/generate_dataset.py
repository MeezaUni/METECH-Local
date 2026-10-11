"""Generate the training data for the KMRL depot copilot.

There is no public corpus of KMRL logbooks, so we synthesise one. Each example is
built from a structured ground truth (the updates or query) and then *rendered*
into the messy text depot staff actually write: WhatsApp shorthand, shift
handover notes, abbreviations, typos, Malayalam-English code-mixing, several
facts per message, and plenty of chatter that contains no update at all.

Because the text is rendered from the answer, every label is correct by
construction. Diversity comes from many templates x many surface forms x noise.

    uv run python ../ml/generate_dataset.py            # from backend/
    -> ml/data/train.jsonl, ml/data/val.jsonl
       (plus teacher rewrites from ml/data/paraphrased.jsonl when paraphrase.py has been run)
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
import sys
import zlib
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.ai.local_format import ask_messages, dump_intent, dump_updates, parse_messages  # noqa: E402
from app.priority import apply_policy  # noqa: E402

FLEET_MAX = 40  # trained on up to 40 sets: KMRL's fleet is growing past today's 25


# ------------------------------------------------------------- surface forms
def train_ref(n: int, rng: random.Random) -> str:
    forms = [
        f"TS-{n:02d}", f"TS-{n}", f"TS {n}", f"TS{n:02d}", f"ts-{n:02d}", f"ts {n}",
        f"train {n}", f"Train {n}", f"train no. {n}", f"Train No {n}", f"train #{n}",
        f"rake {n:02d}", f"Rake {n}", f"set {n}", f"Set-{n:02d}", f"KMRL-TS-{n:03d}",
        f"trainset {n}", f"unit {n}", f"TS-{n:03d}", f"T{n:02d}",
    ]
    weights = [9, 6, 6, 3, 3, 2, 7, 4, 2, 2, 2, 4, 3, 2, 1, 4, 2, 1, 3, 1]
    return rng.choices(forms, weights)[0]


def tname(n: int) -> str:
    return f"KMRL-TS-{n:03d}"


DEPT_FORMS = {
    "Rolling-Stock": ["rolling stock", "Rolling Stock", "Rolling-Stock", "RS", "rolling-stock", "R/S"],
    "Signalling": ["signalling", "Signalling", "signaling", "signal", "Signal", "S&T"],
    "Telecom": ["telecom", "Telecom", "telecommunication", "comms", "Telecom dept", "TC"],
}
CERT_WORDS = ["cert", "certificate", "fitness certificate", "FC", "fitness cert", "fitness", "certificate of fitness"]

MONTHS = list(calendar.month_name)[1:]
MON_ABBR = list(calendar.month_abbr)[1:]


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def render_expiry(today: date, rng: random.Random) -> tuple[str, date]:
    """Return (phrase, resolved date). Mix of absolute and relative forms."""
    kind = rng.random()
    if kind < 0.28:  # relative
        choice = rng.choice(["months", "year", "days", "weeks", "fortnight"])
        if choice == "fortnight":
            return rng.choice(["for a fortnight", "for one fortnight", "for another fortnight"]), today + timedelta(days=14)
        if choice == "months":
            n = rng.choice([1, 2, 3, 6, 6, 9, 12])
            word = rng.choice([str(n), {1: "one", 2: "two", 3: "three", 6: "six", 12: "twelve"}.get(n, str(n))])
            return rng.choice([f"for {word} months", f"for another {word} months", f"for {word} month" if n == 1 else f"for {word} months"]), add_months(today, n)
        if choice == "year":
            return rng.choice(["for one year", "for a year", "for 1 year", "for another year"]), add_months(today, 12)
        if choice == "days":
            n = rng.choice([15, 30, 45, 60, 90, 120, 180])
            return f"for {n} days", today + timedelta(days=n)
        n = rng.choice([2, 3, 4, 6, 8])
        return f"for {n} weeks", today + timedelta(weeks=n)
    d = today + timedelta(days=rng.randint(10, 540))
    fmt = rng.choice(["iso", "dmy_long", "mdy_long", "slash", "dash", "ord_abbr", "abbr", "no_year"])
    if fmt == "iso":
        return d.isoformat(), d
    if fmt == "dmy_long":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}", d
    if fmt == "mdy_long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}", d
    if fmt == "slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}", d
    if fmt == "dash":
        return f"{d.day}-{d.month}-{d.year}", d
    if fmt == "ord_abbr":
        return f"{ordinal(d.day)} {MON_ABBR[d.month - 1]} {d.year}", d
    if fmt == "abbr":
        return f"{d.day} {MON_ABBR[d.month - 1]} {d.year}", d
    # Day + month without a year means the next such date from today.
    d2 = date(today.year, d.month, min(d.day, 28))
    if d2 <= today:
        d2 = date(today.year + 1, d2.month, d2.day)
    return rng.choice([f"{d2.day} {MONTHS[d2.month - 1]}", f"{MON_ABBR[d2.month - 1]} {d2.day}", f"{ordinal(d2.day)} {MONTHS[d2.month - 1]}"]), d2


# ----------------------------------------------------------- defect lexicon
DEFECTS = {
    "Critical": [
        "wheel flat detected", "brake cylinder pressure drop", "crack on bogie frame", "CBTC onboard unit fault",
        "door obstruction detection not working", "smoke from under-frame", "emergency brake not releasing",
        "ATP failure", "brake failure on car 3", "axle bearing overheating", "wheel flange damage",
        "door opening on wrong side", "parking brake stuck applied", "fire alarm triggered in car 1",
        "brake pads worn out", "abnormal vibration from the bogie", "burning smell from car 2", "sparks seen at the collector",
        "door opened while running", "suspension air bag deflated",
    ],
    "High": [
        "HVAC not cooling in car 2", "PIS display dead", "CCTV offline in car 3", "door 3 not closing",
        "pantograph uplift pressure low", "traction motor overheating", "radio not working", "headlight not working",
        "saloon AC tripping", "PA announcements not audible", "door 2 failed to open", "auxiliary converter fault",
        "CCTV feed lost in car 1", "cab camera offline", "destination board not updating", "PIS screen not working",
        "no cooling in car 3", "AC not working in car 4", "door 1 stuck closed", "intercom not working", "head lamp fused",
    ],
    "Routine": [
        "door 3 slow to close", "saloon light flickering", "seat panel damaged", "grab pole loose",
        "minor HVAC noise", "wiper blade worn", "floor mat torn", "window glass scratched",
        "announcement volume low", "route map sticker peeling", "handrail cover loose", "cab seat adjuster stiff",
        "interior panel rattling", "LED strip partly off in car 4",
        "door 2 slow to open", "seat cushion torn", "door window glass scratched", "AC making a humming noise",
        "minor rattle in car 2", "floor tile chipped", "cracked window pane in car 3",
    ],
}
FIXED_PHRASES = ["fixed", "rectified", "attended", "repaired", "sorted", "replaced", "done"]
NOISE_PREFIX = ["", "", "", "", "[22:41] ", "[23:05] ", "Shift handover: ", "Night shift update - ", "FYI ", "Hi sir, ",
                "Update: ", "Depot log: ", "Good evening. ", "SSE update: ", "Noted - ", "pls note: "]
NOISE_SUFFIX = ["", "", "", "", " - Ramesh SSE", " Thanks", " pls note", " ok", " -Anil", " (night shift)", " sheri",
                " ok aanu", " Regards, Depot control", " 👍"]
FILLERS = {"completed": ["completed", "kazhinju", "done", "over", "finished"], "done": ["done", "cheythu", "completed", "ok"]}


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 5 or not word.isalpha():
        return word
    i = rng.randint(1, len(word) - 2)
    op = rng.choice(["drop", "swap", "double"])
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def roughen(text: str, rng: random.Random, protect: list[str]) -> str:
    """Casual-writing noise. Never touches train refs, numbers, dates or departments in `protect`."""
    if rng.random() < 0.25:
        text = text.lower()
        protect = [p.lower() for p in protect]
    if rng.random() < 0.25:
        words = text.split(" ")
        for i, w in enumerate(words):
            if rng.random() < 0.12 and not any(ch.isdigit() for ch in w) and not any(w in p or p in w for p in protect):
                words[i] = typo(w, rng)
        text = " ".join(words)
    return text


# ------------------------------------------------------------ fact renderers
def fact_renew(n: int, today: date, rng: random.Random) -> tuple[str, list[dict]]:
    tr = train_ref(n, rng)
    cw = rng.choice(CERT_WORDS)
    if rng.random() < 0.15:  # two departments in one sentence
        d1, d2 = rng.sample(list(DEPT_FORMS), 2)
        phrase, when = render_expiry(today, rng)
        f1, f2 = rng.choice(DEPT_FORMS[d1]), rng.choice(DEPT_FORMS[d2])
        if phrase.startswith("for "):
            text = rng.choice([f"{f1} & {f2} {cw} for {tr} extended {phrase}", f"{tr} {f1} and {f2} {cw}s renewed {phrase}"])
        else:
            text = rng.choice([
                f"{tr} {f1} and {f2} {cw}s renewed till {phrase}",
                f"{f1} & {f2} {cw} for {tr} extended upto {phrase}",
                f"Both {f1} and {f2} {cw}s of {tr} renewed, valid until {phrase}",
            ])
        ups = [{"train_name": tname(n), "action": "renew_certificate", "department": d, "date": when.isoformat()} for d in (d1, d2)]
        return text, ups, [tr, f1, f2, phrase]
    dept = rng.choice(list(DEPT_FORMS))
    df = rng.choice(DEPT_FORMS[dept])
    phrase, when = render_expiry(today, rng)
    if phrase.startswith("for "):
        text = rng.choice([
            f"{df} {cw} of {tr} extended {phrase}",
            f"{tr} {df} {cw} renewed {phrase}",
            f"{df} dept renewed {cw} for {tr} {phrase}",
            f"{tr}: {df} {cw} validity extended {phrase}",
        ])
    else:
        text = rng.choice([
            f"{df} {cw} for {tr} renewed till {phrase}",
            f"{tr} {df} {cw} renewed, valid until {phrase}",
            f"Received new {df} {cw} for {tr}, expiry {phrase}",
            f"{tr}: {df} fitness ok upto {phrase}",
            f"{df} dept issued {cw} to {tr} valid till {phrase}",
            f"{cw} ({df}) {tr} renewed - valid up to {phrase}",
            f"{tr} got {df} {cw} renewed. New expiry {phrase}",
            f"renewed {df} {cw} {tr} till {phrase}",
        ])
    return text, [{"train_name": tname(n), "action": "renew_certificate", "department": dept, "date": when.isoformat()}], [tr, df, phrase]


def fact_close(n: int, today: date, rng: random.Random) -> tuple[str, list[dict], list[str]]:
    if rng.random() < 0.15:  # several job cards closed in one sentence
        ids = rng.sample(range(1001, 1251), rng.choice([2, 2, 3]))
        refs = [rng.choice([f"JC {i}", f"job card {i}", f"WO {i}", f"#{i}"]) for i in ids]
        listed = ", ".join(refs[:-1]) + rng.choice([" and ", " & ", ", "]) + refs[-1]
        text = rng.choice([f"{listed} closed", f"Closed {listed}", f"{listed} all completed", f"{listed} both closed" if len(ids) == 2 else f"{listed} all closed"])
        return text, [{"train_name": "", "action": "close_job_card", "job_card_id": i} for i in ids], refs
    jid = rng.randint(1001, 1250)
    jc = rng.choice([f"job card #{jid}", f"JC {jid}", f"JC#{jid}", f"job card {jid}", f"work order {jid}", f"WO-{jid}", f"job {jid}", f"JC-{jid}"])
    with_train = rng.random() < 0.65
    tr = train_ref(n, rng)
    defect = rng.choice(DEFECTS[rng.choice(["Critical", "High", "Routine"])])
    fx = rng.choice(FIXED_PHRASES)
    if with_train:
        text = rng.choice([
            f"{tr} {defect} {fx}, closed {jc}",
            f"Closed {jc} on {tr}",
            f"{jc} for {tr} closed",
            f"{tr}: {jc} completed",
            f"{jc} ({tr}) attended and closed",
            f"{tr} {jc} resolved",
            f"closed {jc} - {tr} {defect} {fx}",
        ])
        train = tname(n)
    else:
        text = rng.choice([f"Closed {jc}", f"{jc} closed", f"{jc} completed", f"{jc} done", f"{jc} attended and closed", f"{jc} resolved, pls update"])
        train = ""
    return text, [{"train_name": train, "action": "close_job_card", "job_card_id": jid}], [tr, jc]


def fact_open(n: int, today: date, rng: random.Random) -> tuple[str, list[dict], list[str]]:
    tr = train_ref(n, rng)
    priority = rng.choices(["Critical", "High", "Routine"], [3, 3, 4])[0]
    defect = rng.choice(DEFECTS[priority])
    explicit = ""
    r = rng.random()
    if r < 0.12:
        explicit = rng.choice([" - urgent", ", critical", " URGENT", ", safety issue", ", withdraw immediately"])
    elif r < 0.20:
        explicit = rng.choice([" (minor)", ", low priority", " - not urgent", ", raise routine JC", " - minor issue", " (cosmetic)"])
    text = rng.choice([
        f"{tr} {defect}{explicit}",
        f"{defect} on {tr}{explicit}",
        f"{defect} reported in {tr}{explicit}",
        f"{tr}: {defect}, pls raise JC{explicit}",
        f"Raise job card for {tr} - {defect}{explicit}",
        f"{tr} driver reported {defect}{explicit}",
        f"Noticed {defect} in {tr}{explicit}",
        f"{tr} - {defect}. Need attention{explicit}",
        f"Defect: {tr} {defect}{explicit}",
    ])
    desc = defect[0].upper() + defect[1:]
    # The depot policy (app/priority.py) labels priority, exactly as the backend applies it.
    priority = apply_policy(desc, text, priority)[0]
    # Urgency words are protected from the typo noise so the text keeps saying what the label means.
    return text, [{"train_name": tname(n), "action": "open_job_card", "description": desc, "priority": priority}], [tr, defect, *explicit.strip(" ,-()").split()]


def fact_flag_clean(n: int, today: date, rng: random.Random) -> tuple[str, list[dict], list[str]]:
    tr = train_ref(n, rng)
    k = rng.randint(1, 4)
    text = rng.choice([
        f"{tr} needs a deep clean", f"Graffiti on {tr} car {k}", f"Vomit in {tr}, needs cleaning",
        f"{tr} very dirty, pls schedule deep cleaning", f"Spill in coach {k} of {tr}", f"{tr} interior stained, deep clean required",
        f"Vandalism reported on {tr}", f"{tr} requires deep cleaning", f"pls put {tr} for deep cleaning tonight",
        f"{tr} floor sticky, cleaning required", f"Paint sprayed on {tr} car {k}",
    ])
    return text, [{"train_name": tname(n), "action": "flag_cleaning"}], [tr]


def fact_cleaned(n: int, today: date, rng: random.Random) -> tuple[str, list[dict], list[str]]:
    tr = train_ref(n, rng)
    done = rng.choice(FILLERS["done"])
    text = rng.choice([
        f"{tr} deep cleaned", f"Deep cleaning of {tr} {done}", f"{tr} cleaning completed", f"{tr} washed and detailed",
        f"{tr} cleaning {done}", f"deep clean {tr} {done}", f"{tr} interior detailing completed", f"Cleaning gang finished {tr}",
    ])
    return text, [{"train_name": tname(n), "action": "mark_cleaned"}], [tr]


def fact_maint(n: int, today: date, rng: random.Random) -> tuple[str, list[dict], list[str]]:
    tr = train_ref(n, rng)
    done = rng.choice(FILLERS["completed"])
    text = rng.choice([
        f"{tr} A-check {done}", f"{tr} inspection done", f"IBL work on {tr} finished", f"{tr} B-check {done}",
        f"{tr} out of IBL after inspection", f"Maintenance completed on {tr}", f"{tr} scheduled inspection {done}",
        f"Inspection of {tr} carried out, all ok", f"{tr} periodic maintenance {done}",
    ])
    return text, [{"train_name": tname(n), "action": "complete_maintenance"}], [tr]


def fact_km(n: int, today: date, rng: random.Random) -> tuple[str, list[dict], list[str]]:
    tr = train_ref(n, rng)
    km = rng.randint(150, 7500)
    kms = rng.choice([f"{km} km", f"{km:,} km", f"{km}km", f"{km} kms", f"{km} kilometres"])
    text = rng.choice([
        f"{tr} has run {kms} since last inspection", f"{tr} odometer since A-check: {kms}", f"{tr} clocked {kms} since inspection",
        f"km since last check for {tr} = {kms}", f"{tr} mileage since IBL {kms}", f"{tr} covered {kms} since last maintenance",
    ])
    return text, [{"train_name": tname(n), "action": "update_mileage", "km": km}], [tr, kms]


FACTS = [(fact_renew, 22), (fact_close, 16), (fact_open, 22), (fact_flag_clean, 10), (fact_cleaned, 10), (fact_maint, 10), (fact_km, 6)]

CHATTER = [
    "Good night all", "Who has the IBL key?", "Meeting at 10 tomorrow in the depot office", "Rain expected tonight, yard drains checked",
    "Canteen closed tomorrow", "Shift change at 22:00 as usual", "All trains stabled", "Pls acknowledge", "Ok noted",
    "Tea break 15 mins", "Depot control: radio check done", "Tomorrow's roster shared in group", "Security round completed",
    "Power block on line 3 from 01:00 to 04:00", "Thanks everyone", "Visitor pass for contractor issued",
    "TS-05 looks fine today", "No issues reported on train 12", "Train 8 stabled at SL2", "Checked TS-14, nothing to report",
    "Will update after inspection", "Waiting for the signalling team", "Any update on the spares?",
]


def parse_example(today: date, rng: random.Random) -> dict:
    if rng.random() < 0.08:
        text = rng.choice(CHATTER)
        if rng.random() < 0.3:
            text += ". " + rng.choice(CHATTER)
        return {"messages": parse_messages(text, today), "target": dump_updates([]), "task": "parse"}

    k = rng.choices([1, 2, 3, 4], [45, 30, 17, 8])[0]
    trains = rng.sample(range(1, FLEET_MAX + 1), k)
    parts: list[str] = []
    updates: list[dict] = []
    for n in trains:
        fn = rng.choices([f for f, _ in FACTS], [w for _, w in FACTS])[0]
        clause, ups, protect = fn(n, today, rng)
        clause = roughen(clause, rng, protect)
        for u in ups:
            u["source_text"] = clause
        parts.append(clause)
        updates.extend(ups)
        if rng.random() < 0.08:  # chatter mixed into real updates
            parts.append(rng.choice(CHATTER[:16]))

    joiner = rng.choice([". ", ".\n", "\n", "; ", ". ", ".  "])
    body = joiner.join(parts)
    if rng.random() < 0.5 and not body.endswith("."):
        body += "."
    text = rng.choice(NOISE_PREFIX) + body + rng.choice(NOISE_SUFFIX)
    return {"messages": parse_messages(text, today), "target": dump_updates(updates), "task": "parse"}


# ------------------------------------------------------------------ questions
STATUS_WORDS = {
    "Service": ["in service", "running tomorrow", "inducted", "going out in the morning", "in revenue service"],
    "Standby": ["on standby", "spare", "in reserve", "standby"],
    "Inspection Bay": ["in the inspection bay", "in IBL", "going for inspection", "in maintenance tonight"],
    "Cleaning": ["being cleaned", "in the cleaning bay", "going for deep cleaning"],
    "Held": ["held", "on hold", "held back"],
    "Unfit": ["unfit", "red", "grounded", "not fit", "out of service"],
    "Warning": ["yellow", "on warning", "flagged", "amber"],
    "Fit": ["fit", "green", "healthy", "fully fit"],
}


def ask_example(rng: random.Random) -> dict:
    intent = rng.choices(
        ["count", "list", "train_detail", "explain_assignment", "expiring_certificates", "open_jobs",
         "branding_status", "plan_summary", "highest_wear", "help"],
        [12, 12, 12, 12, 10, 10, 7, 8, 7, 10],
    )[0]
    out: dict = {"intent": intent}
    if intent in ("count", "list"):
        status = rng.choice(list(STATUS_WORDS))
        w = rng.choice(STATUS_WORDS[status])
        noun = rng.choice(["trains", "rakes", "trainsets", "sets"])
        out["status"] = status
        q = rng.choice([
            f"How many {noun} are {w}?", f"number of {noun} {w}", f"how many {w}", f"count of {noun} {w} tomorrow",
            f"Total {noun} {w}?", f"how many {noun} {w} tonight",
        ] if intent == "count" else [
            f"Which {noun} are {w}?", f"list {noun} {w}", f"show me {noun} {w}", f"names of {noun} {w}",
            f"which ones are {w}", f"{noun} {w}?", f"give me the {w} list",
        ])
    elif intent in ("train_detail", "explain_assignment"):
        n = rng.randint(1, FLEET_MAX)
        tr = train_ref(n, rng)
        out["train_name"] = tname(n)
        q = rng.choice([
            f"Tell me about {tr}", f"status of {tr}", f"{tr} details", f"what's up with {tr}", f"give me info on {tr}",
            f"{tr}?", f"show {tr}", f"how is {tr} doing", f"anything wrong with {tr}?", f"issues with {tr}",
        ] if intent == "train_detail" else [
            f"Why is {tr} on standby?", f"why was {tr} sent to IBL", f"reason {tr} is held", f"explain {tr}'s assignment",
            f"why isn't {tr} in service", f"why is {tr} going for cleaning", f"why did the planner pick {tr}", f"how come {tr} is not running",
        ])
    elif intent == "expiring_certificates":
        days = rng.choice([3, 5, 7, 10, 14, 15, 21, 30])
        mode = rng.choices(["days", "week", "fortnight", "soon"], [60, 15, 15, 10])[0]
        out["days"] = {"days": days, "week": 7, "fortnight": 14, "soon": 7}[mode]
        cert = rng.choice(["certificates", "certs", "fitness certificates", "FCs"])
        verb = rng.choice(["expire", "lapse", "run out", "are due"])
        if mode == "week":
            q = rng.choice([f"{cert} expiring this week", f"any {cert} due this week", f"{cert} that {verb} within a week"])
        elif mode == "fortnight":
            q = rng.choice([f"which {cert} {verb} in the next fortnight", f"{cert} expiring within a fortnight", f"any {cert} running out this fortnight"])
        elif mode == "soon":
            q = rng.choice([f"any {cert} expiring soon?", f"which {cert} {verb} soon", f"{cert} about to lapse"])
        else:
            q = rng.choice([
                f"Which {cert} {verb} in {days} days?", f"{cert} expiring in next {days} days", f"{cert} due within {days} days",
                f"upcoming cert expiries within {days} days", f"any {cert} that {verb} in the coming {days} days",
            ])
    elif intent == "open_jobs":
        pr = rng.choice([None, None, "Critical", "High", "Routine"])
        if pr:
            out["priority"] = pr
        p = pr.lower() + " " if pr else ""
        q = rng.choice([f"show open {p}job cards", f"open {p}jobs", f"pending {p}work orders", f"any {p}defects pending",
                        f"list open {p}JCs", f"what {p}maintenance is pending", f"outstanding {p}work", f"{p}jobs still open"])
    elif intent == "branding_status":
        q = rng.choice(["How are the branding contracts doing?", "advertising exposure status", "which ads are behind", "branding SLA report",
                        "are we meeting advertiser hours", "wrap contract status"])
    elif intent == "plan_summary":
        q = rng.choice(["summary of tonight's plan", "give me the plan", "overview for tomorrow", "what's the induction plan",
                        "plan summary pls", "how does tomorrow look"])
    elif intent == "highest_wear":
        n = rng.choice([None, 3, 5, 10])
        if n:
            out["limit"] = n
        q = rng.choice([f"top {n} trains by km since inspection", f"top {n} highest wear trains"]) if n else rng.choice([
            "which trains have the highest wear", "most used trains", "highest mileage trains", "which rakes are due for inspection soon by km"])
    else:
        q = rng.choice(["hello", "what can you do?", "who are you", "tell me a joke", "weather in kochi", "help", "thanks", "good morning"])
    if rng.random() < 0.2:
        q = q.lower()
    return {"messages": ask_messages(q), "target": dump_intent(out), "task": "ask"}


# ---------------------------------------------------------------------- main
def build(n_parse: int, n_ask: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_parse):
        today = date(2025, 6, 1) + timedelta(days=rng.randint(0, 900))
        rows.append(parse_example(today, rng))
    rows += [ask_example(rng) for _ in range(n_ask)]
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parse", type=int, default=6000)
    ap.add_argument("--train-ask", type=int, default=2000)
    ap.add_argument("--val-parse", type=int, default=400)
    ap.add_argument("--val-ask", type=int, default=150)
    args = ap.parse_args()

    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    sets = {"train": build(args.train_parse, args.train_ask, seed=1), "val": build(args.val_parse, args.val_ask, seed=99)}

    # Teacher rewrites from paraphrase.py, if present. Split by base message so rewrites of one
    # message never land in both train and val.
    teacher = out / "paraphrased.jsonl"
    if teacher.exists():
        added = {"train": 0, "val": 0}
        for line in open(teacher, encoding="utf-8"):
            r = json.loads(line)
            split = "val" if zlib.crc32(r["base_id"].encode()) % 10 == 0 else "train"
            sets[split].append({"messages": r["messages"], "target": r["target"], "task": r["task"]})
            added[split] += 1
        random.Random(5).shuffle(sets["train"])
        print(f"teacher rewrites added: {added['train']} train, {added['val']} val")

    for name, rows in sets.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> {out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
