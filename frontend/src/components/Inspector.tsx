import { useState } from 'react'
import type { Decision, Department, FleetUpdate, Plan, PlannerConfig, Priority, TrainView } from '../api'
import { DEPARTMENTS, daysBetween, fmtDate, fmtKm, shortName } from '../api'
import { lampClass } from './PlanViews'

interface Props {
  train: TrainView | null
  decision: Decision | null
  was: string | null
  drafted: boolean
  plan: Plan
  cfg: PlannerConfig
  today: string
  onClose: () => void
  onDraft: (u: FleetUpdate) => void
}

const SCORE_LABELS: [keyof Decision['score'], string][] = [
  ['branding', 'Branding'], ['mileage', 'Wear'], ['shunting', 'Shunting'],
  ['risk', 'Cert risk'], ['maintenance', 'Open work'], ['cleaning', 'Cleaning'],
]

export function Inspector({ train, decision, was, drafted, plan, cfg, today, onClose, onDraft }: Props) {
  if (!train || !decision) return <PlanExplainer plan={plan} cfg={cfg} />
  const open = train.job_cards.filter((j) => j.status === 'Open')
  const scores = SCORE_LABELS.filter(([k]) => decision.score[k] !== 0)
  const maxScore = Math.max(1, ...scores.map(([k]) => Math.abs(decision.score[k])))

  return (
    <div>
      <div className="inspector__head">
        <h2 className="inspector__name">{shortName(train.name)}</h2>
        <button className="inspector__close" onClick={onClose} aria-label="Close train details">×</button>
      </div>
      <p className="badge">
        <span className={lampClass(decision.assignment)}>{decision.assignment}</span>
        {was && <span className="chip__moved">(was {was})</span>}
        <span><span className={`readiness readiness--${decision.readiness}`} style={{ marginLeft: 6, marginRight: 6 }} />{decision.readiness}</span>
      </p>
      {drafted && <p className="muted" style={{ marginTop: 6 }}>Includes your draft changes. The records below show saved data.</p>}

      <section className="section">
        <h4>Why this assignment</h4>
        <ul className="reasons">
          {decision.reasons.map((r) => (
            <li key={r} className={r.startsWith('Watch:') ? 'reasons--watch' : ''}>{r.replace(/^Watch: /, '')}</li>
          ))}
        </ul>
      </section>

      {scores.length > 0 && (
        <section className="section">
          <h4>What weighed on the decision</h4>
          <div className="score">
            {scores.map(([k, label]) => {
              const v = decision.score[k]
              const w = (Math.abs(v) / maxScore) * 50
              return (
                <div key={k} className="score__row">
                  <span>{label}</span>
                  <span className="score__track"><span className={`score__bar score__bar--${v >= 0 ? 'pos' : 'neg'}`} style={{ width: `${w}%` }} /></span>
                  <span className="score__val">{v > 0 ? `+${v}` : v}</span>
                </div>
              )
            })}
          </div>
        </section>
      )}

      <section className="section">
        <h4>Train</h4>
        <dl className="facts">
          <dt>Stabled</dt><dd>{train.stabling_track}, slot {train.stabling_position}{decision.shunting_moves ? `, ${decision.shunting_moves} shunt${decision.shunting_moves > 1 ? 's' : ''} to exit` : ''}</dd>
          <dt>Since inspection</dt><dd>{fmtKm(train.km_since_maintenance)} of {fmtKm(cfg.maint_limit_km)}</dd>
          <dt>Lifetime</dt><dd>{fmtKm(train.total_km)}</dd>
          <dt>In service since</dt><dd>{fmtDate(train.commissioned, { month: 'short', year: 'numeric' })}</dd>
          <dt>Last deep clean</dt><dd>{fmtDate(train.last_deep_clean)} ({daysBetween(train.last_deep_clean, today)} days ago){train.cleaning_flagged ? ', flagged' : ''}</dd>
        </dl>
      </section>

      <section className="section">
        <h4>Fitness certificates</h4>
        <ul className="item-list">
          {train.certificates.map((c) => {
            const d = daysBetween(today, c.expires)
            return (
              <li key={c.id} className={`item ${d < 0 ? 'item--bad' : d <= cfg.cert_warning_days ? 'item--warn' : ''}`}>
                <span>{c.department}<br /><span className="item__sub">{d < 0 ? `Expired ${-d} day${d === -1 ? '' : 's'} ago` : `Valid ${d} more day${d === 1 ? '' : 's'}`}</span></span>
                <span>{fmtDate(c.expires, { day: 'numeric', month: 'short', year: 'numeric' })}</span>
              </li>
            )
          })}
        </ul>
      </section>

      <section className="section">
        <h4>Open job cards</h4>
        {open.length === 0 ? <p className="muted">No open job cards.</p> : (
          <ul className="item-list">
            {open.map((j) => (
              <li key={j.id} className={`item ${j.priority === 'Critical' ? 'item--bad' : 'item--warn'}`}>
                <span>#{j.id} {j.description}<br /><span className="item__sub">{j.priority}, {j.system}, opened {fmtDate(j.opened)}</span></span>
                <button className="btn btn--small" onClick={() => onDraft({ train_name: train.name, action: 'close_job_card', job_card_id: j.id })}>Close</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {train.branding && (
        <section className="section">
          <h4>Branding</h4>
          <p>{train.branding.advertiser}</p>
          <div className="progress" aria-hidden>
            <span className="progress__fill" style={{ width: `${Math.min(100, (train.branding.delivered_hours / train.branding.target_hours) * 100)}%` }} />
            <span className="progress__mark" style={{ left: `${Math.min(100, (train.branding.expected_hours / train.branding.target_hours) * 100)}%` }} />
          </div>
          <p className="muted">
            {Math.round(train.branding.delivered_hours)} of {train.branding.target_hours} contracted hours,{' '}
            {train.branding.deficit_hours > 0 ? `${Math.round(train.branding.deficit_hours)} h behind schedule` : 'on schedule'}, {train.branding.days_left} days left
            {train.branding.at_risk ? '. At risk of missing the contract.' : ''}
          </p>
        </section>
      )}

      <WhatIf train={train} today={today} onDraft={onDraft} />
    </div>
  )
}

function WhatIf({ train, today, onDraft }: { train: TrainView; today: string; onDraft: (u: FleetUpdate) => void }) {
  const [dept, setDept] = useState<Department>(train.certificates.find((c) => c.expires < today)?.department ?? 'Rolling-Stock')
  const plusYear = new Date(today + 'T00:00:00')
  plusYear.setFullYear(plusYear.getFullYear() + 1)
  const [until, setUntil] = useState(plusYear.toISOString().slice(0, 10))
  const [desc, setDesc] = useState('')
  const [priority, setPriority] = useState<Priority>('Routine')
  const n = train.name

  return (
    <section className="section">
      <h4>Try a change</h4>
      <p className="muted" style={{ marginBottom: 10 }}>Changes go to the draft first, so you can see the new plan before saving.</p>
      <div className="actions">
        <div className="action-row">
          <button className="btn btn--small" onClick={() => onDraft({ train_name: n, action: 'mark_cleaned' })}>Deep clean done</button>
          <button className="btn btn--small" onClick={() => onDraft({ train_name: n, action: 'flag_cleaning' })}>Needs deep clean</button>
          <button className="btn btn--small" onClick={() => onDraft({ train_name: n, action: 'complete_maintenance' })}>Inspection done</button>
        </div>
        <div className="action-row">
          <label className="field">Certificate
            <select className="select" value={dept} onChange={(e) => setDept(e.target.value as Department)}>
              {DEPARTMENTS.map((d) => <option key={d}>{d}</option>)}
            </select>
          </label>
          <label className="field">Valid until
            <input className="date" type="date" value={until} min={today} onChange={(e) => setUntil(e.target.value)} />
          </label>
          <button className="btn btn--small" style={{ alignSelf: 'end' }} onClick={() => onDraft({ train_name: n, action: 'renew_certificate', department: dept, date: until })}>Renew</button>
        </div>
        <form
          className="action-row"
          onSubmit={(e) => { e.preventDefault(); if (desc.trim()) { onDraft({ train_name: n, action: 'open_job_card', description: desc.trim(), priority }); setDesc('') } }}
        >
          <input className="input" placeholder="Report a defect, e.g. door 3 slow to close" value={desc} onChange={(e) => setDesc(e.target.value)} aria-label="Defect description" />
          <select className="select" value={priority} onChange={(e) => setPriority(e.target.value as Priority)} aria-label="Priority">
            <option>Routine</option><option>High</option><option>Critical</option>
          </select>
          <button className="btn btn--small" type="submit" disabled={!desc.trim()}>Open job card</button>
        </form>
      </div>
    </section>
  )
}

function PlanExplainer({ plan, cfg }: { plan: Plan; cfg: PlannerConfig }) {
  const k = plan.kpis
  return (
    <div className="inspector__empty">
      <h2 style={{ fontSize: 24, color: 'var(--ivory)' }}>How tonight's plan was made</h2>
      <p>Select a train on the depot or in the plan to see why it got its assignment.</p>
      <p>
        The planner gives every trainset exactly one place for tomorrow. It never inducts a train with an expired
        certificate, an open critical job card or more than {fmtKm(cfg.maint_limit_km)} since inspection. Within those
        rules it fills {cfg.service_target} timetable slots and keeps {cfg.min_standby} on standby, then prefers:
      </p>
      <ul className="reasons">
        <li>branded trains that are behind on contracted exposure hours</li>
        <li>trains with less wear since their last inspection</li>
        <li>trains parked where they can leave without shunting</li>
        <li>using the {cfg.ibl_slots} inspection bays and {cfg.cleaning_slots} cleaning bays for the most urgent work</li>
      </ul>
      <p className="muted">
        Inducted trains average {fmtKm(k.avg_service_wear_km)} since inspection, against {fmtKm(k.avg_fleet_wear_km)} across all fit trains.
        {' '}{k.branding_trains_in_service} of {k.branding_trains_total} branded trains run tomorrow.
      </p>
    </div>
  )
}
