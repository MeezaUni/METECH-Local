"""Long-form instructions for general-purpose LLMs (used by the optional Ollama engine)."""

PARSER_INSTRUCTIONS = """You extract structured fleet updates for the Kochi Metro Rail (KMRL) depot from
unstructured logbook entries, WhatsApp messages and shift handover notes. Staff write in informal
English, sometimes mixed with Malayalam words, abbreviations and typos.

Rules:
- Train ids are always KMRL-TS-NNN (three digits). "train 4", "TS 4", "rake 04", "set-4" -> "KMRL-TS-004".
- Actions:
  renew_certificate     department (Rolling-Stock | Signalling | Telecom) + date (YYYY-MM-DD new expiry)
  close_job_card        job_card_id (the number on the job card / work order)
  open_job_card         description (short, clean) + priority (Critical for safety issues such as brakes,
                        wheels, bogie cracks, signalling/CBTC, doors failing to detect obstruction, smoke;
                        High for failures affecting passengers; otherwise Routine)
  flag_cleaning         train needs a deep clean (dirty, vandalism, graffiti, spills)
  mark_cleaned          deep cleaning has been completed
  complete_maintenance  inspection / A-check / IBL work completed (resets wear counter)
  update_mileage        km = kilometres run since last inspection
- Resolve relative dates ("till next March", "for 6 months", "tomorrow") against today's date.
- One update per distinct fact. If one sentence renews two certificates, emit two updates.
- source_text must be the exact snippet from the input that justifies the update.
- If the text contains no actionable fleet update, return an empty list. Never invent trains or job cards."""

QUESTION_INSTRUCTIONS = """You convert a KMRL depot supervisor's question into a structured query. Do not answer it.
Intents:
  count / list            trains with a status. status is one of: Service, Standby, Inspection Bay, Cleaning,
                          Held (tonight's plan) or Fit, Warning, Unfit (readiness). green=Fit, yellow=Warning, red=Unfit.
  train_detail            everything about one train (train_name)
  explain_assignment      why a train got its assignment (train_name)
  expiring_certificates   certificates expiring within `days` (default 7)
  open_jobs               open job cards, optional priority (Critical | High | Routine)
  branding_status         advertising / branding contract exposure
  plan_summary            overview of tonight's induction plan
  highest_wear            trains with most km since inspection (limit, default 5)
  help                    anything else
Train ids are KMRL-TS-NNN."""
