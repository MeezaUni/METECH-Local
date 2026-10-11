"""Synthetic but realistic KMRL fleet data.

Everything is generated relative to `today` from a fixed seed, so a fresh database
always contains the same mix of situations the planner has to handle: a couple of
expired fitness certificates, critical job cards, trains overdue for inspection or
deep cleaning, and branding contracts that are behind on exposure hours.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from .models import DEPARTMENTS, BrandingContract, Certificate, Fleet, JobCard, Train

NUM_TRAINS = 25
TRACKS = ("SL1", "SL2", "SL3", "SL4", "SL5")
SERVICE_START = date(2017, 6, 17)  # Kochi Metro opened to the public

SYSTEMS = {
    "Brakes": ["Brake pad wear beyond limit", "Brake cylinder pressure drop", "Parking brake not releasing"],
    "HVAC": ["HVAC coolant leak in car 2", "Saloon temperature above setpoint", "HVAC compressor noise"],
    "Doors": ["Door sensor calibration", "Door obstruction detection fault", "Door closing time out of range"],
    "Bogie": ["Bogie structural crack inspection", "Wheel flat detected", "Wheel profile out of tolerance"],
    "Pantograph": ["Pantograph carbon strip worn", "Pantograph uplift pressure low"],
    "Telecom": ["PIS display not updating", "CCTV camera offline in car 3", "Radio antenna repair"],
    "Signalling": ["CBTC onboard unit fault", "ATP balise reader intermittent"],
    "Interior": ["Saloon lighting replacement", "Seat panel damaged", "Grab pole loose"],
}
CRITICAL_ISSUES = {
    "Brakes": "Brake cylinder pressure drop",
    "Bogie": "Wheel flat detected",
    "Signalling": "CBTC onboard unit fault",
    "Doors": "Door obstruction detection fault",
}
ADVERTISERS = [
    "Spice Coast Bank", "Periyar Silks", "Marine Drive Mall", "Vembanad Resorts",
    "Kochi Tech Park", "Backwater Ayurveda", "Malabar Spice Co.", "Chendamangalam Handlooms",
]


def generate_fleet(today: date | None = None, seed: int = 2025) -> Fleet:
    today = today or date.today()
    rng = random.Random(seed)

    trains: list[Train] = []
    for i in range(1, NUM_TRAINS + 1):
        # The first 15 sets arrived for the 2017 opening, the rest later.
        commissioned = SERVICE_START + timedelta(days=rng.randint(-60, 30)) if i <= 15 else date(2019, 9, 1) + timedelta(days=rng.randint(0, 500))
        days = (today - commissioned).days
        total_km = int(days * 0.92 * 225 * rng.uniform(0.9, 1.1))
        km_since = rng.randint(400, 5600)
        trains.append(Train(
            id=i,
            name=f"KMRL-TS-{i:03d}",
            commissioned=commissioned,
            total_km=total_km,
            km_since_maintenance=km_since,
            last_deep_clean=today - timedelta(days=rng.randint(1, 9)),
            stabling_track=TRACKS[(i - 1) // 5],
            stabling_position=(i - 1) % 5 + 1,
        ))

    # Hand-placed situations so every seeded night has something interesting.
    trains[6].km_since_maintenance = 5940      # TS-007 right at the inspection limit
    trains[11].km_since_maintenance = 5120     # TS-012 approaching it
    trains[2].last_deep_clean = today - timedelta(days=13)   # TS-003 overdue clean
    trains[16].last_deep_clean = today - timedelta(days=12)  # TS-017 overdue clean
    trains[20].cleaning_flagged = True                       # TS-021 vandalism reported

    certificates: list[Certificate] = []
    cid = 1
    for t in trains:
        for dept in DEPARTMENTS:
            issued = today - timedelta(days=rng.randint(20, 300))
            expires = today + timedelta(days=rng.randint(15, 330))
            certificates.append(Certificate(id=cid, train_id=t.id, department=dept, issued=issued, expires=expires))
            cid += 1

    def cert(train_no: int, dept: str) -> Certificate:
        return next(c for c in certificates if c.train_id == train_no and c.department == dept)

    cert(5, "Telecom").expires = today - timedelta(days=2)          # TS-005 expired
    cert(14, "Rolling-Stock").expires = today - timedelta(days=1)   # TS-014 expired
    cert(9, "Signalling").expires = today + timedelta(days=1)       # TS-009 expiring
    cert(22, "Telecom").expires = today + timedelta(days=2)         # TS-022 expiring

    job_cards: list[JobCard] = []
    jid = 1001
    for t in trains:
        for _ in range(rng.randint(2, 4)):
            system = rng.choice(list(SYSTEMS))
            opened = today - timedelta(days=rng.randint(20, 300))
            job_cards.append(JobCard(
                id=jid, train_id=t.id, system=system, description=rng.choice(SYSTEMS[system]),
                priority=rng.choice(["High", "Routine", "Routine"]), status="Closed",
                opened=opened, closed=opened + timedelta(days=rng.randint(1, 4)),
            ))
            jid += 1

    def open_job(train_no: int, system: str, priority: str, description: str | None = None) -> None:
        nonlocal jid
        job_cards.append(JobCard(
            id=jid, train_id=train_no, system=system,
            description=description or rng.choice(SYSTEMS[system]), priority=priority,
            status="Open", opened=today - timedelta(days=rng.randint(0, 6)),
        ))
        jid += 1

    open_job(11, "Bogie", "Critical", CRITICAL_ISSUES["Bogie"])
    open_job(19, "Brakes", "Critical", CRITICAL_ISSUES["Brakes"])
    open_job(2, "HVAC", "High")
    open_job(8, "Doors", "Routine")
    open_job(16, "Telecom", "Routine")
    open_job(23, "Interior", "Routine")
    open_job(12, "Pantograph", "High")

    branding: list[BrandingContract] = []
    branded_trains = [4, 6, 10, 13, 18, 20, 24, 25]
    for k, (train_no, advertiser) in enumerate(zip(branded_trains, ADVERTISERS), start=1):
        period_start = today - timedelta(days=rng.randint(8, 22))
        period_end = period_start + timedelta(days=30)
        target = rng.choice([300, 340, 380])
        elapsed = (today - period_start).days
        pro_rata = target * elapsed / 30
        delivered = round(pro_rata * rng.uniform(0.72, 1.08), 1)
        branding.append(BrandingContract(
            id=k, train_id=train_no, advertiser=advertiser, target_hours=target,
            delivered_hours=delivered, period_start=period_start, period_end=period_end,
        ))

    return Fleet(trains=trains, certificates=certificates, job_cards=job_cards, branding=branding)
