import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import type { AIInfo, FleetUpdate, FleetView, Plan, Simulation } from './api'
import { api, fmtDate } from './api'
import { AskPanel, HistoryPanel, LogbookPanel } from './components/AIPanels'
import { Inspector } from './components/Inspector'
import { DraftTray, FleetTable, PlanBoard, Tally, lampClass } from './components/PlanViews'
import type { DraftItem } from './components/PlanViews'

// three.js is most of the bundle; load the depot view separately so the rest of the page paints first.
const Depot3D = lazy(() => import('./components/Depot3D').then((m) => ({ default: m.Depot3D })))

type Tab = 'plan' | 'fleet' | 'logbook' | 'ask' | 'history'
const TABS: [Tab, string][] = [['plan', 'Plan'], ['fleet', 'Fleet'], ['logbook', 'Logbook'], ['ask', 'Ask'], ['history', 'History']]

interface Toast { text: string; error?: boolean }

export default function App() {
  const [fleet, setFleet] = useState<FleetView | null>(null)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [ai, setAi] = useState<AIInfo | null>(null)
  const [draft, setDraft] = useState<DraftItem[]>([])
  const [sim, setSim] = useState<Simulation | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('plan')
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<Toast | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [historyKey, setHistoryKey] = useState(0)

  const notify = useCallback((text: string, error = false) => {
    setToast({ text, error })
    window.setTimeout(() => setToast((t) => (t?.text === text ? null : t)), 4500)
  }, [])

  const reload = useCallback(async () => {
    const [f, p] = await Promise.all([api.fleet(), api.plan()])
    setFleet(f); setPlan(p); setHistoryKey((k) => k + 1)
  }, [])

  useEffect(() => {
    reload().catch((e) => setLoadError((e as Error).message))
    api.aiInfo().then(setAi).catch(() => setAi(null))
  }, [reload])

  // Every draft change re-runs the planner on a copy of the fleet.
  useEffect(() => {
    if (!draft.length) { setSim(null); return }
    let cancelled = false
    const timer = window.setTimeout(() => {
      api.simulate(draft.map((d) => d.update))
        .then((s) => { if (!cancelled) setSim(s) })
        .catch((e) => notify(`Simulation failed: ${(e as Error).message}`, true))
    }, 150)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [draft, notify])

  const addDraft = useCallback((updates: FleetUpdate[], via: 'manual' | 'ai') => {
    setDraft((d) => [...d, ...updates.map((update) => ({ update, via }))])
    setTab((t) => (t === 'logbook' ? 'plan' : t))
  }, [])

  async function save(items: DraftItem[]) {
    setBusy(true)
    try {
      const applied: string[] = []
      const rejected: string[] = []
      for (const via of ['manual', 'ai'] as const) {
        const group = items.filter((i) => i.via === via).map((i) => i.update)
        if (!group.length) continue
        const r = await api.apply(group, via)
        applied.push(...r.applied); rejected.push(...r.rejected)
      }
      await reload()
      setDraft([])
      if (rejected.length) notify(`Saved ${applied.length}, skipped ${rejected.length}: ${rejected[0]}`, true)
      else notify(`Saved ${applied.length} change${applied.length === 1 ? '' : 's'} to the fleet`)
    } catch (e) {
      notify(`Could not save: ${(e as Error).message}`, true)
    } finally {
      setBusy(false)
    }
  }

  async function approve() {
    setBusy(true)
    try {
      const r = await api.approve('Depot supervisor')
      setHistoryKey((k) => k + 1)
      notify(`Plan #${r.id} for ${fmtDate(r.plan_date, { weekday: 'long', day: 'numeric', month: 'short' })} approved`)
    } catch (e) {
      notify(`Could not approve: ${(e as Error).message}`, true)
    } finally {
      setBusy(false)
    }
  }

  const shown = sim?.simulated ?? plan
  const changes = useMemo(() => sim?.changes ?? [], [sim])
  const changedSet = useMemo(() => new Set(changes.map((c) => c.train_name)), [changes])

  if (loadError) {
    return (
      <div className="splash">
        <h1>The planner backend is not reachable</h1>
        <p>Start it with <code>run.bat</code> (or <code>uv run uvicorn app.main:app --port 8020</code> in the backend folder), then reload this page.</p>
        <p className="muted">{loadError}</p>
      </div>
    )
  }
  if (!fleet || !plan || !shown) {
    return <div className="splash"><h1>Loading the depot</h1><p>Reading fleet data and solving tonight's plan.</p></div>
  }

  const train = fleet.trains.find((t) => t.name === selected) ?? null
  const decision = shown.decisions.find((d) => d.train_name === selected) ?? null
  const was = changes.find((c) => c.train_name === selected)?.before ?? null
  const openJobs = fleet.trains.reduce((n, t) => n + t.job_cards.filter((j) => j.status === 'Open').length, 0)

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="7" fill="#1b2e39" />
            <path d="M5 22h22M5 16h22M5 10h22" stroke="#e6e1d3" strokeWidth="1.4" opacity=".7" />
            <rect x="9" y="13.6" width="12" height="4.8" rx="1.6" fill="#1aa39c" />
          </svg>
          <div>
            <h1 className="topbar__title">Muttom depot planner</h1>
            <p className="topbar__sub">Induction plan for {fmtDate(shown.plan_date, { weekday: 'long', day: 'numeric', month: 'long' })}</p>
          </div>
        </div>
        {ai && (
          <span className={`engine ${ai.ready ? 'engine--ready' : ''}`} title={ai.note || `${ai.engine} (${ai.model})`}>
            <span className="engine__lamp" />
            {ai.ready ? `${ai.engine}, ${ai.model}` : ai.note || `${ai.engine} not ready`}
          </span>
        )}
        <button
          className="btn btn--primary"
          onClick={approve}
          disabled={busy || draft.length > 0}
          title={draft.length ? 'Save or discard the draft changes first' : 'Record this plan as approved'}
        >
          Approve plan
        </button>
      </header>

      <main className="main">
        <div className="stage">
          <div className="depot">
            <Suspense fallback={<p className="splash">Drawing the depot…</p>}>
              <Depot3D decisions={shown.decisions} selected={selected} changed={changedSet} onSelect={setSelected} />
            </Suspense>
            <div className="depot__legend" aria-hidden>
              {(['Service', 'Standby', 'Inspection Bay', 'Cleaning', 'Held'] as const).map((a) => <span key={a} className={lampClass(a)}>{a}</span>)}
            </div>
            <p className="depot__hint">Drag to orbit, scroll to zoom, click a train</p>
          </div>
          <Tally plan={shown} baseline={sim ? plan : null} />
          <DraftTray
            items={draft}
            changes={changes}
            warnings={sim?.warnings ?? []}
            busy={busy}
            onRemove={(i) => setDraft((d) => d.filter((_, j) => j !== i))}
            onDiscard={() => setDraft([])}
            onSave={() => save(draft)}
          />
          <nav className="tabs" role="tablist" aria-label="Workspace">
            {TABS.map(([key, label]) => (
              <button key={key} role="tab" className="tab" aria-selected={tab === key} onClick={() => setTab(key)}>
                {label}
                {key === 'fleet' && <span className="tab__count">{fleet.trains.length}</span>}
                {key === 'plan' && changes.length > 0 && <span className="tab__count">{changes.length} changed</span>}
                {key === 'history' && <span className="tab__count">{openJobs} open jobs</span>}
              </button>
            ))}
          </nav>
          <div className="workspace" role="tabpanel">
            {tab === 'plan' && <PlanBoard plan={shown} changes={changes} selected={selected} onSelect={setSelected} />}
            {tab === 'fleet' && <FleetTable trains={fleet.trains} plan={shown} today={fleet.today} cfg={fleet.planner} selected={selected} onSelect={setSelected} />}
            {tab === 'logbook' && (
              <LogbookPanel
                ai={ai}
                onDraft={(u) => addDraft(u, 'ai')}
                onSaveNow={(u) => save(u.map((update) => ({ update, via: 'ai' as const })))}
              />
            )}
            {tab === 'ask' && <AskPanel />}
            {tab === 'history' && (
              <HistoryPanel
                refreshKey={historyKey}
                onReset={async () => { await api.reset(); setDraft([]); setSelected(null); await reload(); notify('Demo data regenerated') }}
              />
            )}
          </div>
        </div>
        <aside className="inspector" aria-label="Train details">
          <Inspector
            train={train}
            decision={decision}
            was={was}
            drafted={draft.some((d) => d.update.train_name === selected)}
            plan={shown}
            cfg={fleet.planner}
            today={fleet.today}
            onClose={() => setSelected(null)}
            onDraft={(u) => { addDraft([u], 'manual'); notify('Added to draft') }}
          />
        </aside>
      </main>
      {toast && <div className={`toast ${toast.error ? 'toast--error' : ''}`} role="status">{toast.text}</div>}
    </div>
  )
}
