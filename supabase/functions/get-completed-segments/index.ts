import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// Inline CORS headers
const corsHeaders = {
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  try {
    const body = await req.json();
    const { run_id, project_id } = body;

    if (!run_id) {
      throw new Error("Missing run_id");
    }

    const authHeader = req.headers.get("Authorization")?.replace("Bearer ", "");

    const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
    const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
    const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

    let supabase;
    if (authHeader === SUPABASE_SERVICE_ROLE_KEY) {
      supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
    } else if (authHeader) {
      supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        global: { headers: { Authorization: `Bearer ${authHeader}` } },
      });
      const { data: { user }, error } = await supabase.auth.getUser();
      if (error || !user) {
        throw new Error("Invalid authentication");
      }
    } else {
      throw new Error("Missing Authorization");
    }

    let query = supabase.from("tasks").select("params, output_location").eq("task_type", "travel_segment").eq("status", "Complete");

    if (authHeader === SUPABASE_SERVICE_ROLE_KEY) {
      // Bypass RLS for service role
      query = query // Supabase JS doesn't have direct usingServiceRole, but since it's created with service key, it bypasses RLS
    } else if (project_id) {
      query = query.eq("project_id", project_id);
    }

    const { data, error } = await query;

    if (error) throw error;

    const results: { segment_index: number; output_location: string }[] = [];
    for (const row of data || []) {
      const params = typeof row.params === "string" ? JSON.parse(row.params) : row.params;
      if (params.orchestrator_run_id === run_id && typeof params.segment_index === "number" && row.output_location) {
        results.push({ segment_index: params.segment_index, output_location: row.output_location });
      }
    }

    results.sort((a, b) => a.segment_index - b.segment_index);

    return new Response(JSON.stringify(results), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error(error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
}); 