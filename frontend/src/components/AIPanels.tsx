import { useEffect, useRef, useState } from 'react'
import type { AIInfo, ApprovedPlan, AuditEntry, FleetUpdate, ParseResponse } from '../api'
import { api, describeUpdate, fmtDate } from '../api'

const LOGBOOK_EXAMPLES = [
  'Telecom cert for TS-05 renewed till 30 Sept 2027. Closed job card #1070, wheel reprofiled.',
  'Night shift: train 3 deep cleaned. TS 12 needs a deep clean, graffiti on car 2.',
  'TS-19 brake cylinder fixed, closed JC 1071. TS-07 A-check completed. Rake 22 door 3 slow to close.',
  'Signalling FC for train 9 extended for 6 months. TS-16 PIS display not working.',
]

const QUESTION_EXAMPLES = [
  'How many trains are in service tomorrow?',
  'Why is TS-09 on standby?',
  'Which certificates expire in the next 7 days?',
  'Show open critical job cards',
  'How are the branding contracts doing?',
  'Which trains have the highest wear?',
]

// ------------------------------------------------------------------ logbook
export function LogbookPanel({ ai, onDraft, onSaveNow }: {
  ai: AIInfo | null
  onDraft: (updates: FleetUpdate[]) => void
  onSaveNow: (updates: FleetUpdate[]) => Promise<void>
}) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ParseResponse | null>(null)
  const [checked, setChecked] = useState<boolean[]>([])
  const [error, setError] = useState<string | null>(null)

  async function read() {
    setBusy(true); setError(null); setResult(null)
    try {
      const r = await api.parse(text)
      setResult(r)
      setChecked(r.updates.map(() => true))
    } catch (e) {
      setError(`Could not read the logbook: ${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const picked = result ? result.updates.filter((_, i) => checked[i]) : []
  const done = () => { setResult(null); setText('') }

  return (
    <div>
      <p className="panel-intro">
        Paste shift handover notes, logbook lines or WhatsApp messages. {ai?.edition === 'local' ? 'The on-device model' : 'The AI'} turns
        them into fleet updates. You check each one before anything changes.
      </p>
      <textarea
        className="textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. Rolling stock cert for TS-14 renewed until 12 Nov 2027. Train 21 deep cleaned."
        aria-label="Logbook text"
      />
      <div className="examples">
        <span className="muted" style={{ alignSelf: 'center' }}>Try:</span>
        {LOGBOOK_EXAMPLES.map((ex) => <button key={ex} className="example" onClick={() => setText(ex)}>{ex}</button>)}
      </div>
      <div className="row">
        <button className="btn btn--primary" onClick={read} disabled={busy || !text.trim()}>{busy ? 'Reading…' : 'Read logbook'}</button>
        {error && <span className="warnings">{error}</span>}
      </div>

      {result && (
        <div className="parsed">
          <p className="muted">Read by {result.engine}. {result.updates.length ? 'Untick anything that looks wrong.' : ''}</p>
          {result.updates.length === 0 && <p>No fleet updates found in that text. Mention a train and what happened to it.</p>}
          {result.updates.map((u, i) => (
            <label key={i} className="parsed__item">
              <input type="checkbox" checked={!!checked[i]} onChange={() => setChecked(checked.map((c, j) => (j === i ? !c : c)))} />
              <span>{describeUpdate(u)}</span>
              {u.source_text && <span className="parsed__quote">“{u.source_text}”</span>}
            </label>
          ))}
          {result.warnings.length > 0 && <div className="warnings">{result.warnings.map((w) => <span key={w}>{w}</span>)}</div>}
          {result.updates.length > 0 && (
            <div className="row" style={{ marginTop: 6 }}>
              <button className="btn" disabled={!picked.length} onClick={() => { onDraft(picked); done() }}>Add {picked.length} to draft</button>
              <button className="btn btn--primary" disabled={!picked.length} onClick={async () => { await onSaveNow(picked); done() }}>Save {picked.length} to fleet</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ ask
interface Msg { q: string; a?: string; engine?: string; error?: boolean }

export function AskPanel() {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }) }, [msgs])

  async function ask(question: string) {
    if (!question.trim()) return
    setBusy(true); setQ('')
    setMsgs((m) => [...m, { q: question }])
    try {
      const r = await api.ask(question)
      setMsgs((m) => m.map((x, i) => (i === m.length - 1 ? { ...x, a: r.answer, engine: r.engine } : x)))
    } catch (e) {
      setMsgs((m) => m.map((x, i) => (i === m.length - 1 ? { ...x, a: `Could not answer: ${(e as Error).message}`, error: true } : x)))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <p className="panel-intro">Ask about the fleet or tonight's plan. Answers are computed from live depot data, so the numbers are always exact.</p>
      {msgs.length > 0 && (
        <div className="chat" aria-live="polite">
          {msgs.map((m, i) => (
            <div key={i} style={{ display: 'contents' }}>
              <p className="msg msg--q">{m.q}</p>
              {m.a ? <p className="msg msg--a">{m.a}{m.engine && <span className="msg__engine">Understood by {m.engine}</span>}</p>
                : <p className="msg msg--a muted">Checking the depot…</p>}
            </div>
          ))}
          <div ref={end} />
        </div>
      )}
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); ask(q) }}>
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question" aria-label="Question" />
        <button className="btn btn--primary" type="submit" disabled={busy || !q.trim()}>Ask</button>
      </form>
      <div className="examples">
        {QUESTION_EXAMPLES.map((ex) => <button key={ex} className="example" onClick={() => ask(ex)} disabled={busy}>{ex}</button>)}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ history
export function HistoryPanel({ refreshKey, onReset }: { refreshKey: number; onReset: () => void }) {
  const [audit, setAudit] = useState<AuditEntry[]>([])
  const [plans, setPlans] = useState<ApprovedPlan[]>([])
  useEffect(() => {
    api.audit().then(setAudit).catch(() => setAudit([]))
    api.history().then(setPlans).catch(() => setPlans([]))
  }, [refreshKey])

  return (
    <div className="history">
      <section>
        <h3>Approved plans</h3>
        {plans.length === 0 ? <p className="muted">No plan approved yet. Approve tonight's plan from the top bar once it looks right.</p> : (
          <div className="log">
            {plans.map((p) => (
              <div key={p.id} className="log__row">
                <span className="log__ts">{fmtDate(p.approved_at.slice(0, 10))} {p.approved_at.slice(11, 16)}</span>
                <span className="log__via">#{p.id}</span>
                <span>For {fmtDate(p.plan_date, { weekday: 'short', day: 'numeric', month: 'short' })}: {p.kpis.service} service, {p.kpis.standby} standby, {p.kpis.inspection} inspection, {p.kpis.cleaning} cleaning, {p.kpis.held} held. Approved by {p.approved_by}.</span>
              </div>
            ))}
          </div>
        )}
      </section>
      <section>
        <h3>Change log</h3>
        <div className="log">
          {audit.map((a) => (
            <div key={a.id} className="log__row">
              <span className="log__ts">{fmtDate(a.ts.slice(0, 10))} {a.ts.slice(11, 16)}</span>
              <span className="log__via">{a.via === 'ai' ? 'logbook' : a.via}</span>
              <span>{a.detail}</span>
            </div>
          ))}
        </div>
      </section>
      <section>
        <h3>Demo data</h3>
        <p className="muted" style={{ marginBottom: 10 }}>Regenerate the 25-trainset demo fleet. This clears the change log and approved plans.</p>
        <button className="btn" onClick={() => { if (confirm('Reset the fleet to fresh demo data? The change log and approved plans will be cleared.')) onReset() }}>Reset demo data</button>
      </section>
    </div>
  )
}
