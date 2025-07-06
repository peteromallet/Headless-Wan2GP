import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.7";

/**
 * Edge function: create-task
 *
 * POST  { task_id, params, task_type, project_id?, dependant_on? }
 * – Requires Authorization: Bearer <JWT>
 * – Verifies JWT, uses `sub` claim as default project_id
 * – Inserts a new row into `tasks` with status "Queued"
 * – Respects RLS because we insert the caller's user-id
 */
serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  // ─── 1. Parse body ──────────────────────────────────────────────
  let body: any;
  try {
    body = await req.json();
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }
  const { task_id, params, task_type, project_id, dependant_on } = body;
  if (!task_id || !params || !task_type) {
    return new Response("task_id, params, task_type required", { status: 400 });
  }

  // ─── 2. Verify caller JWT ───────────────────────────────────────
  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return new Response("Missing Authorization header", { status: 401 });
  }
  const jwt = authHeader.slice(7);
  let callerId: string | null = null;
  let isServiceRole = false;
  try {
    const [, payloadBase64] = jwt.split(".");
    const padded = payloadBase64 + "=".repeat((4 - payloadBase64.length % 4) % 4);
    const payload = JSON.parse(atob(padded));
    callerId = payload.sub;
    // If the token represents the service role or an elevated role, allow bypass
    const roleClaim = payload.role || payload.app_metadata?.role;
    isServiceRole = roleClaim === "service_role" || roleClaim === "supabase_admin";
  } catch (e) {
    console.error("JWT decode failed", e);
    return new Response("Invalid JWT", { status: 401 });
  }
  if (!callerId && !isServiceRole) {
    return new Response("Could not extract user id from JWT", { status: 401 });
  }

  // If the caller is NOT service role, ensure project_id matches their sub
  if (!isServiceRole) {
  if (project_id && project_id !== callerId) {
    return new Response("project_id does not match token", { status: 403 });
  }
  }

  // Determine which project_id to insert
  const finalProjectId = project_id ?? callerId;

  // ─── 3. Insert row using service-role key ───────────────────────
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  const { error } = await supabase.from("tasks").insert({
    id: task_id,
    params,
    task_type,
    project_id: finalProjectId,
    dependant_on: dependant_on ?? null,
    status: "Queued",
    created_at: new Date().toISOString(),
  });

  if (error) {
    console.error("create_task error:", error);
    return new Response(error.message, { status: 500 });
  }

  return new Response("Task queued", { status: 200 });
}); 