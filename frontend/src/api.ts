// Typed client for the FastAPI backend. Shapes mirror backend/app/models.py.

export type Department = 'Rolling-Stock' | 'Signalling' | 'Telecom'
export type Priority = 'Critical' | 'High' | 'Routine'
export type Readiness = 'Fit' | 'Warning' | 'Unfit'
export type Assignment = 'Service' | 'Standby' | 'Inspection Bay' | 'Cleaning' | 'Held'

export const ASSIGNMENTS: Assignment[] = ['Service', 'Standby', 'Inspection Bay', 'Cleaning', 'Held']
export const DEPARTMENTS: Department[] = ['Rolling-Stock', 'Signalling', 'Telecom']

export interface Certificate { id: number; train_id: number; department: Department; issued: string; expires: string }
export interface JobCard {
  id: number; train_id: number; system: string; description: string; priority: Priority
  status: 'Open' | 'Closed'; opened: string; closed: string | null
}
export interface Branding {
  id: number; train_id: number; advertiser: string; target_hours: number; delivered_hours: number
  period_start: string; period_end: string; expected_hours: number; deficit_hours: number; days_left: number; at_risk: boolean
}
export interface TrainView {
  id: number; name: string; commissioned: string; total_km: number; km_since_maintenance: number
  last_deep_clean: string; cleaning_flagged: boolean; stabling_track: string; stabling_position: number
  readiness: Readiness; blockers: string[]; warnings: string[]
  certificates: Certificate[]; job_cards: JobCard[]; branding: Branding | null
}
export interface PlannerConfig {
  service_target: number; min_standby: number; ibl_slots: number; cleaning_slots: number
  cert_warning_days: number; cleaning_interval_days: number; maint_warning_km: number; maint_limit_km: number
}
export interface FleetView { today: string; trains: TrainView[]; planner: PlannerConfig }

export interface ScoreBreakdown { branding: number; mileage: number; shunting: number; maintenance: number; cleaning: number; risk: number }
export interface Decision {
  train_id: number; train_name: string; assignment: Assignment; readiness: Readiness; reasons: string[]
  score: ScoreBreakdown; stabling_track: string; stabling_position: number; shunting_moves: number
  km_since_maintenance: number; total_km: number
}
export interface PlanKPIs {
  service: number; service_target: number; standby: number; inspection: number; cleaning: number; held: number
  shunting_moves: number; branding_trains_in_service: number; branding_trains_total: number; branding_at_risk: string[]
  avg_service_wear_km: number; avg_fleet_wear_km: number; solve_ms: number; solver_status: string
}
export interface Plan { generated_at: string; plan_date: string; decisions: Decision[]; kpis: PlanKPIs; alerts: string[] }
export interface PlanChange { train_name: string; before: Assignment; after: Assignment }
export interface Simulation { baseline: Plan; simulated: Plan; changes: PlanChange[]; applied: string[]; warnings: string[] }

export type UpdateAction =
  | 'renew_certificate' | 'close_job_card' | 'open_job_card' | 'flag_cleaning'
  | 'mark_cleaned' | 'complete_maintenance' | 'update_mileage'

export interface FleetUpdate {
  train_name: string; action: UpdateAction; department?: Department | null; date?: string | null
  job_card_id?: number | null; description?: string | null; priority?: Priority | null; km?: number | null
  source_text?: string | null
}
export interface ParseResponse { updates: FleetUpdate[]; warnings: string[]; engine: string }
export interface AskResponse { answer: string; intent: Record<string, unknown>; engine: string }
export interface ApplyResult { applied: string[]; rejected: string[] }
export interface AuditEntry { id: number; ts: string; actor: string; via: string; action: string; detail: string }
export interface AIInfo { edition: 'cloud' | 'local'; engine: string; model: string; ready: boolean; note: string }
export interface ApprovedPlan { id: number; plan_date: string; approved_at: string; approved_by: string; kpis: PlanKPIs }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try { detail = (await res.json()).detail ?? detail } catch { /* not JSON */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json() as Promise<T>
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  fleet: () => request<FleetView>('/fleet'),
  plan: () => request<Plan>('/plan'),
  simulate: (updates: FleetUpdate[]) => post<Simulation>('/plan/simulate', { updates }),
  apply: (updates: FleetUpdate[], via: 'manual' | 'ai') => post<ApplyResult>('/fleet/updates', { updates, via }),
  approve: (approved_by: string) => post<{ id: number; plan_date: string }>('/plan/approve', { approved_by }),
  history: () => request<ApprovedPlan[]>('/plan/history'),
  audit: () => request<AuditEntry[]>('/audit'),
  aiInfo: () => request<AIInfo>('/ai/info'),
  parse: (text: string) => post<ParseResponse>('/ai/parse', { text }),
  ask: (question: string) => post<AskResponse>('/ai/ask', { question }),
  reset: () => post<{ status: string }>('/admin/reset', {}),
}

// ------------------------------------------------------------ formatting
export const shortName = (name: string) => name.replace('KMRL-', '')

export function describeUpdate(u: FleetUpdate): string {
  const t = shortName(u.train_name)
  switch (u.action) {
    case 'renew_certificate': return `${t}: renew ${u.department} certificate until ${u.date ?? '?'}`
    case 'close_job_card': return `${t}: close job card #${u.job_card_id}`
    case 'open_job_card': return `${t}: open ${u.priority ?? 'Routine'} job card — ${u.description}`
    case 'flag_cleaning': return `${t}: flag for deep cleaning`
    case 'mark_cleaned': return `${t}: deep clean completed`
    case 'complete_maintenance': return `${t}: inspection completed`
    case 'update_mileage': return `${t}: ${u.km?.toLocaleString('en-IN')} km since inspection`
  }
}

export const fmtKm = (km: number) => `${km.toLocaleString('en-IN')} km`

export function fmtDate(iso: string, opts: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short' }) {
  return new Date(iso + (iso.length === 10 ? 'T00:00:00' : '')).toLocaleDateString('en-IN', opts)
}

export function daysBetween(fromIso: string, toIso: string) {
  return Math.round((new Date(toIso + 'T00:00:00').getTime() - new Date(fromIso + 'T00:00:00').getTime()) / 86_400_000)
}
