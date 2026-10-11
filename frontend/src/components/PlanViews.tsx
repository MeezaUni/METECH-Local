import { useMemo, useState } from 'react'
import type { Assignment, Decision, FleetUpdate, Plan, PlanChange, PlannerConfig, TrainView } from '../api'
import { ASSIGNMENTS, daysBetween, describeUpdate, fmtDate, fmtKm, shortName } from '../api'

export const lampClass = (a: Assignment) => `lamp lamp--${a.split(' ')[0]}`
const lampVar = (a: Assignment) => `var(--sig-${({ Service: 'service', Standby: 'standby', 'Inspection Bay': 'ibl', Cleaning: 'cleaning', Held: 'held' } as const)[a]})`

const GROUP_NOTE: Record<Assignment, string> = {
  Service: 'leaves the depot for the 05:30 timetable',
  Standby: 'fit and ready to replace a failed train',
  'Inspection Bay': 'maintenance tonight on the inspection bay line',
  Cleaning: 'deep cleaning tonight',
  Held: 'unfit, waiting for an inspection bay slot',
}

// ------------------------------------------------------------------ tally
export function Tally({ plan, baseline }: { plan: Plan; baseline: Plan | null }) {
  const k = plan.kpis
  const b = baseline?.kpis
  const delta = (now: number, before?: number) => {
    if (before === undefined || now === before) return null
    const d = now - before
    return <span className="tally__delta" title="Change from the saved plan">{d > 0 ? `+${d}` : `−${-d}`}</span>
  }
  return (
    <div className="tally" aria-label="Plan totals">
      <span className="tally__item"><span className={lampClass('Service')}>Service</span>
        <span className="tally__value">{k.service}</span><span className="tally__of">of {k.service_target}</span>{delta(k.service, b?.service)}</span>
      <span className="tally__item"><span className={lampClass('Standby')}>Standby</span><span className="tally__value">{k.standby}</span>{delta(k.standby, b?.standby)}</span>
      <span className="tally__item"><span className={lampClass('Inspection Bay')}>Inspection bay</span><span className="tally__value">{k.inspection}</span>{delta(k.inspection, b?.inspection)}</span>
      <span className="tally__item"><span className={lampClass('Cleaning')}>Cleaning</span><span className="tally__value">{k.cleaning}</span>{delta(k.cleaning, b?.cleaning)}</span>
      <span className="tally__item"><span className={lampClass('Held')}>Held</span><span className="tally__value">{k.held}</span>{delta(k.held, b?.held)}</span>
      <span className="tally__item">Shunting moves <span className="tally__value">{k.shunting_moves}</span>{delta(k.shunting_moves, b?.shunting_moves)}</span>
      <span className="tally__meta">Solved in {k.solve_ms} ms ({k.solver_status.toLowerCase()})</span>
    </div>
  )
}

// ------------------------------------------------------------------ draft tray
export interface DraftItem { update: FleetUpdate; via: 'manual' | 'ai' }

interface DraftProps {
  items: DraftItem[]
  changes: PlanChange[]
  warnings: string[]
  busy: boolean
  onRemove: (index: number) => void
  onDiscard: () => void
  onSave: () => void
}

export function DraftTray({ items, changes, warnings, busy, onRemove, onDiscard, onSave }: DraftProps) {
  if (!items.length) return null
  const moved = changes.length
  return (
    <section className="draft" aria-label="Draft changes">
      <div>
        <h2 className="draft__title">{items.length} draft change{items.length > 1 ? 's' : ''}</h2>
        <p className="draft__note">
          {moved === 0 ? 'The plan stays the same with these changes.' : `The plan below shows the result: ${moved} train${moved > 1 ? 's' : ''} change assignment.`}
          {' '}Nothing is saved until you save to the fleet.
        </p>
      </div>
      <div className="draft__actions">
        <button className="btn btn--ghost" onClick={onDiscard} disabled={busy}>Discard</button>
        <button className="btn btn--primary" onClick={onSave} disabled={busy}>Save to fleet</button>
      </div>
      <ul className="draft__list">
        {items.map((it, i) => (
          <li key={i} className="draft__item">
            {describeUpdate(it.update)}
            {it.via === 'ai' && <span className="draft__via">from logbook</span>}
            <button onClick={() => onRemove(i)} aria-label={`Remove: ${describeUpdate(it.update)}`}>×</button>
          </li>
        ))}
      </ul>
      {warnings.map((w) => <p key={w} className="draft__warn">{w}</p>)}
    </section>
  )
}

// ------------------------------------------------------------------ alerts
export function Alerts({ alerts }: { alerts: string[] }) {
  if (!alerts.length) return null
  return (
    <div className="alerts" role="status">
      {alerts.map((a) => (
        <p key={a} className={`alert ${/held|shortfall|expired|below/i.test(a) ? 'alert--stop' : ''}`}>{a}</p>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ plan board
interface BoardProps {
  plan: Plan
  changes: PlanChange[]
  selected: string | null
  onSelect: (name: string) => void
}

export function PlanBoard({ plan, changes, selected, onSelect }: BoardProps) {
  const before = new Map(changes.map((c) => [c.train_name, c.before]))
  return (
    <div className="board">
      <Alerts alerts={plan.alerts} />
      {ASSIGNMENTS.map((a) => {
        const group = plan.decisions.filter((d) => d.assignment === a)
        if (!group.length) return null
        return (
          <section key={a} className="board__group">
            <h3><span className={lampClass(a)}>{a}</span><span className="board__count">{group.length}</span><span className="board__why">{GROUP_NOTE[a]}</span></h3>
            <div className="chips">
              {group.map((d) => <Chip key={d.train_name} d={d} was={before.get(d.train_name)} selected={selected === d.train_name} onSelect={onSelect} />)}
            </div>
          </section>
        )
      })}
    </div>
  )
}

function Chip({ d, was, selected, onSelect }: { d: Decision; was?: Assignment; selected: boolean; onSelect: (n: string) => void }) {
  return (
    <button
      className="chip"
      style={{ ['--lamp' as string]: lampVar(d.assignment) }}
      aria-pressed={selected}
      onClick={() => onSelect(d.train_name)}
      title={d.reasons[0]}
    >
      <span className="chip__name">{shortName(d.train_name)}<span className={`readiness readiness--${d.readiness}`} aria-label={d.readiness} /></span>
      {was ? <span className="chip__moved">was {was}</span> : <span className="chip__meta">{d.stabling_track}, slot {d.stabling_position}</span>}
    </button>
  )
}

// ------------------------------------------------------------------ fleet table
type SortKey = 'name' | 'wear' | 'cert' | 'readiness'

export function FleetTable({ trains, plan, today, cfg, selected, onSelect }: {
  trains: TrainView[]; plan: Plan; today: string; cfg: PlannerConfig; selected: string | null; onSelect: (n: string) => void
}) {
  const [sort, setSort] = useState<SortKey>('name')
  const assignment = new Map(plan.decisions.map((d) => [d.train_name, d.assignment]))
  const rows = useMemo(() => {
    const soonest = (t: TrainView) => Math.min(...t.certificates.map((c) => daysBetween(today, c.expires)))
    const rank = { Unfit: 0, Warning: 1, Fit: 2 }
    const copy = [...trains]
    if (sort === 'wear') copy.sort((a, b) => b.km_since_maintenance - a.km_since_maintenance)
    if (sort === 'cert') copy.sort((a, b) => soonest(a) - soonest(b))
    if (sort === 'readiness') copy.sort((a, b) => rank[a.readiness] - rank[b.readiness])
    return copy
  }, [trains, sort, today])

  const head = (key: SortKey, label: string) => (
    <button onClick={() => setSort(key)} aria-pressed={sort === key}>{label}</button>
  )
  return (
    <div className="table-wrap">
      <table className="fleet">
        <thead>
          <tr>
            <th>{head('name', 'Train')}</th>
            <th>{head('readiness', 'Readiness')}</th>
            <th>Plan</th>
            <th>Stabled</th>
            <th>{head('wear', 'Since inspection')}</th>
            <th>{head('cert', 'Certificates')}</th>
            <th>Open jobs</th>
            <th>Last deep clean</th>
            <th>Branding</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => {
            const a = assignment.get(t.name)!
            const ratio = Math.min(1, t.km_since_maintenance / cfg.maint_limit_km)
            const wearCls = t.km_since_maintenance >= cfg.maint_limit_km ? 'wear__fill--bad' : t.km_since_maintenance >= cfg.maint_warning_km ? 'wear__fill--warn' : ''
            const open = t.job_cards.filter((j) => j.status === 'Open')
            return (
              <tr key={t.id} onClick={() => onSelect(t.name)} aria-selected={selected === t.name}>
                <td className="fleet__name">{shortName(t.name)}</td>
                <td><span className={`readiness readiness--${t.readiness}`} style={{ marginLeft: 0, marginRight: 8 }} />{t.readiness}</td>
                <td><span className={lampClass(a)}>{a}</span></td>
                <td>{t.stabling_track} / {t.stabling_position}</td>
                <td>
                  <span className="wear">
                    <span className="wear__bar"><span className={`wear__fill ${wearCls}`} style={{ width: `${ratio * 100}%` }} /></span>
                    {fmtKm(t.km_since_maintenance)}
                  </span>
                </td>
                <td>
                  <span className="certs" title={t.certificates.map((c) => `${c.department}: ${c.expires}`).join('\n')}>
                    {t.certificates.map((c) => {
                      const d = daysBetween(today, c.expires)
                      return <span key={c.id} className={`cert-pip ${d < 0 ? 'cert-pip--bad' : d <= cfg.cert_warning_days ? 'cert-pip--warn' : ''}`} />
                    })}
                  </span>
                </td>
                <td>{open.length ? open.map((j) => `${j.priority}`).join(', ') : '—'}</td>
                <td>{fmtDate(t.last_deep_clean)}{t.cleaning_flagged ? ' (flagged)' : ''}</td>
                <td>{t.branding ? t.branding.advertiser : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
