# Supabase Edge Function – **claim-next-task**

_Last updated: <!-- KEEP UPDATED --> 2025-07-09_

## 0 ▪︎ Goal
Replace the direct RPC call (`func_claim_task`) in the Python worker with a **Supabase Edge Function** that atomically selects and claims the next task.

Requirements:
1. **Service-role key** (or `supabase_admin`): return the next queued task across **all** users.
2. **User JWT**: return the next queued task **owned by that user only** ( `project_id = auth.uid()` ).
3. Preserve the existing atomic locking semantics (`FOR UPDATE SKIP LOCKED`).
4. Keep backward-compatibility: if the Edge function is unreachable, fall back to the current RPC.

---

## 1 ▪︎ High-Level Steps

| Phase | What | Owner | Est. Time |
|-------|------|-------|-----------|
| 1 | Create Edge function `claim-next-task` | BE | 1-2 h |
| 2 | (Optional) Add new SQL helper `func_claim_user_task` | DBA | 30 m |
| 3 | Add env var `SUPABASE_EDGE_CLAIM_TASK_URL` & update `db_operations.py` | BE | 30 m |
| 4 | Update tests & docs | BE | 1 h |
| 5 | Deploy & verify RLS | DevOps | 30 m |

_Total effort: **~3-4 hours**._

---

## 2 ▪︎ Edge Function Design

### ➤ 2.0  Token Classification & Service-Key Detection  (REVISED)

Logic is now **binary**:

| Case | Condition | Behaviour |
|------|-----------|-----------|
| **Service key** | `token === SUPABASE_SERVICE_ROLE_KEY` **OR** JWT claims `role ∈ {service_role,supabase_admin}` | `isServiceRole = true` → unrestricted queue |
| **User token**  | *all other tokens* (regular JWTs **and** Personal-Access-Tokens) | `isServiceRole = false`; resolve `callerId` via *user_api_token* lookup |

Implementation:

```ts
const AUTH = req.headers.get('Authorization') ?? '';
if (!AUTH.startsWith('Bearer ')) return err401();

const token = AUTH.slice(7);
const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

let callerId: string | null = null;
let isServiceRole = false;

// 1) Hard match against the service-role key string
if (token === serviceKey) {
  isServiceRole = true;
}

// 2) If still undecided, try to decode JWT and inspect role
if (!isServiceRole) {
  try {
    const [, payloadB64] = token.split('.');
    const padded = payloadB64 + '='.repeat((4 - payloadB64.length % 4) % 4);
    const payload: any = JSON.parse(atob(padded));

    if (['service_role', 'supabase_admin'].includes(payload.role ?? payload.app_metadata?.role)) {
      isServiceRole = true;
    } else {
      callerId = payload.sub ?? null;    // may still be null for PAT
    }
  } catch (_) {
    // ignore – likely a PAT
  }
}

// 3) USER TOKEN PATH – we must resolve callerId via user_api_token
if (!isServiceRole) {
  // Every non-service token is treated as a PAT/JWT belonging to a user → look it up
  const { data, error } = await supabaseAdmin
    .from('user_api_token')
    .select('user_id')
    .eq('token', token)         // Option A: raw token column
    // .eq('jti_hash', sha256(token)) // Option B: hashed column
    .single();

  if (error || !data) return err403();
  callerId = data.user_id;
}

// At this point:
//   • if isServiceRole → unrestricted
//   • else           → callerId is a valid user id
```

> **Security**: Prefer storing `jti_hash = SHA-256(token)` in *user_api_token* and compare hashes. Restrict selects on this table to service-role only.

### 2.1  File Location
`supabase/functions/claim_next_task/index.ts`

### 2.2  Request / Response
```
POST /functions/v1/claim-next-task
Headers:  Authorization: Bearer <JWT>
Body   : {} (empty JSON accepted)

200 OK
{
  "task_id":        "...",
  "params":         {...},
  "task_type":      "...",
  "project_id":     "..."
}

204 No Content – no queued tasks
401 / 403 / 500  – error states
```

### 2.3  Logic (pseudo-code)
```ts
const supabaseAdmin = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);
const {jwt, callerId, isServiceRole} = parseAuth(req);
const workerId = crypto.randomUUID();

// Build dynamic SQL
const baseSelect = `
  WITH sel AS (
    SELECT id, task_id, params, task_type, project_id
    FROM tasks
    WHERE status = 'Queued'
      ${!isServiceRole ? "AND project_id = $callerId" : ""}
      AND (dependant_on IS NULL OR dependant_on IN (
            SELECT id FROM tasks WHERE status = 'Complete'))
    ORDER BY created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
  ), upd AS (
    UPDATE tasks
    SET status = 'In Progress', worker_id = $workerId, updated_at = NOW()
    WHERE id = (SELECT id FROM sel)
    RETURNING task_id, params, task_type, project_id
  )
  SELECT * FROM upd;
`;

const { data, error } = await supabaseAdmin.rpc('exec_sql', { sql: baseSelect, ... });
```
*The above can be implemented inline with `supabaseAdmin.rpc('claim_task_sql', { ... })` or by creating `func_claim_user_task` as a proper stored procedure (preferred).*  

If `data.length === 0` → return 204.

If `callerId` is still null after all attempts → return **403 Forbidden**.

---

## 3 ▪︎ SQL Helper Function (optional but cleaner)
Create a variant that accepts `p_user_id`:
```sql
CREATE OR REPLACE FUNCTION func_claim_user_task(
  p_table_name TEXT,
  p_worker_id  TEXT,
  p_user_id    TEXT
) RETURNS TABLE(...) AS $$
BEGIN
  -- Same body as func_claim_task but with
  --   AND project_id = p_user_id
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

Edge function picks which one to call:
```ts
const rpcName = isServiceRole ? 'func_claim_task' : 'func_claim_user_task';
const params   = isServiceRole ?
  { p_table_name: 'tasks', p_worker_id: workerId } :
  { p_table_name: 'tasks', p_worker_id: workerId, p_user_id: callerId };
```

---

## 4 ▪︎ Python Changes (`source/db_operations.py`)
1. **New constant**
```python
SUPABASE_EDGE_CLAIM_TASK_URL: str | None = None  # set by headless.py / env
```
2. **Modify** `get_oldest_queued_task_supabase`:
```
if edge_url := (SUPABASE_EDGE_CLAIM_TASK_URL or os.getenv('SUPABASE_EDGE_CLAIM_TASK_URL')):
    try:
        headers = {'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'}
        resp = httpx.post(edge_url, json={}, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()  # {task_id, params, ...}
        elif resp.status_code == 204:
            return None  # no tasks
        # else fall-through to RPC fallback
    except Exception:
        pass  # fallback
# existing RPC path remains unchanged
```

3. **Thread-safety**: nothing new, still single RPC request per loop.

---

## 5 ▪︎ RLS Policies
```sql
-- Allow owner to select their queued tasks
CREATE POLICY "select_own_queued" ON public.tasks
  FOR SELECT USING (project_id = auth.uid());

-- Allow owner to update their task to 'In Progress'
CREATE POLICY "update_own_to_in_progress" ON public.tasks
  FOR UPDATE USING (project_id = auth.uid())
  WITH CHECK (status = 'In Progress');
```
_Service-role key bypasses RLS._

---

## 6 ▪︎ Test Matrix
| Scenario | Token | Expected |
|----------|-------|----------|
| No tasks | User  | 204 |
| One queued task (user) | Same user | 200 & claimed |
| Task from other user | Different user | 204 |
| Mixed queue, service-role key | Service key | 200 & oldest overall |
| Concurrency (2 workers, user) | User | Each gets a distinct task |

Automate in `test_supabase_headless.py` with pytest-markers `@edge`.

---

## 7 ▪︎ Roll-Out
1. Deploy Edge function to **preview** env → run tests.
2. Update env var in worker containers.
3. Merge to `main` once green.
4. Remove direct RPC call in a later cleanup if Edge path proves stable.

---

## 8 ▪︎ Open Questions
* Do we want to **delete** (vs update) RLS policies around status changes?
* Naming: `claim_next_task` vs `fetch_next_task` – pick one and stay consistent.
* Should we retire `func_claim_task` entirely? (Edge still depends on it unless we inline SQL.) 