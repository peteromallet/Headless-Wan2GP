import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.7";
import { createHash } from "https://deno.land/std@0.224.0/crypto/mod.ts";

/**
 * Edge function: claim-next-task
 * 
 * Claims the next queued task atomically.
 * - Service-role key: claims any task across all users
 * - User token: claims only tasks for that specific user
 * 
 * POST /functions/v1/claim-next-task
 * Headers: Authorization: Bearer <JWT or PAT>
 * Body: {} (empty JSON)
 * 
 * Returns:
 * - 200 OK with task data
 * - 204 No Content if no tasks available
 * - 401 Unauthorized if no valid token
 * - 403 Forbidden if token invalid or user not found
 * - 500 Internal Server Error
 */
serve(async (req) => {
  // Only accept POST requests
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  // Extract authorization header
  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return new Response("Missing or invalid Authorization header", { status: 401 });
  }

  const token = authHeader.slice(7); // Remove "Bearer " prefix
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const supabaseUrl = Deno.env.get("SUPABASE_URL");

  if (!serviceKey || !supabaseUrl) {
    console.error("Missing required environment variables");
    return new Response("Server configuration error", { status: 500 });
  }

  // Create admin client for database operations
  const supabaseAdmin = createClient(supabaseUrl, serviceKey);

  let callerId: string | null = null;
  let isServiceRole = false;

  // 1) Check if token matches service-role key directly
  if (token === serviceKey) {
    isServiceRole = true;
    console.log("Direct service-role key match");
  }

  // 2) If not service key, try to decode as JWT and check role
  if (!isServiceRole) {
    try {
      const parts = token.split(".");
      if (parts.length === 3) {
        // It's a JWT - decode and check role
        const payloadB64 = parts[1];
        const padded = payloadB64 + "=".repeat((4 - (payloadB64.length % 4)) % 4);
        const payload = JSON.parse(atob(padded));

        // Check for service role in various claim locations
        const role = payload.role || payload.app_metadata?.role;
        if (["service_role", "supabase_admin"].includes(role)) {
          isServiceRole = true;
          console.log("JWT has service-role/admin role");
        }
        // Don't extract user ID from JWT - always look it up in user_api_token table
      }
    } catch (e) {
      // Not a valid JWT - will be treated as PAT
      console.log("Token is not a valid JWT, treating as PAT");
    }
  }

  // 3) USER TOKEN PATH - ALWAYS resolve callerId via user_api_token table
  if (!isServiceRole) {
    console.log("Looking up token in user_api_token table...");
    
    try {
      // Query user_api_tokens table to find user
      const { data, error } = await supabaseAdmin
        .from("user_api_tokens")
        .select("user_id")
        .eq("token", token)
        .single();

      if (error || !data) {
        console.error("Token lookup failed:", error);
        return new Response("Invalid or expired token", { status: 403 });
      }

      callerId = data.user_id;
      console.log(`Token resolved to user ID: ${callerId}`);
      
      // Debug: Check user's projects and tasks
      const { data: userProjects } = await supabaseAdmin
        .from("projects")
        .select("id, name")
        .eq("user_id", callerId);
      
      console.log(`DEBUG: User ${callerId} owns ${userProjects?.length || 0} projects`);
      
      if (userProjects && userProjects.length > 0) {
        const projectIds = userProjects.map(p => p.id);
        const { data: userTasks } = await supabaseAdmin
          .from("tasks")
          .select("id, status, project_id, task_type, created_at")
          .in("project_id", projectIds);
        
        console.log(`DEBUG: Found ${userTasks?.length || 0} tasks across user's projects`);
        if (userTasks && userTasks.length > 0) {
          const queuedTasks = userTasks.filter(t => t.status === "Queued");
          console.log(`DEBUG: ${queuedTasks.length} tasks are in 'Queued' status`);
          console.log("DEBUG: Sample tasks:", JSON.stringify(userTasks.slice(0, 3), null, 2));
          
          // Show unique status values to debug enum
          const uniqueStatuses = [...new Set(userTasks.map(t => t.status))];
          console.log(`DEBUG: Unique status values: ${JSON.stringify(uniqueStatuses)}`);
        }
      } else {
        console.log(`DEBUG: User ${callerId} has no projects - cannot claim any tasks`);
      }
    } catch (e) {
      console.error("Error querying user_api_token:", e);
      return new Response("Token validation failed", { status: 403 });
    }
  }

  // Generate a unique worker ID for this request
  const workerId = `edge_${crypto.randomUUID()}`;

  try {
    // Call the appropriate RPC function based on token type
    let rpcResponse;
    
    if (isServiceRole) {
      // Service role: claim any available task (no user restrictions)
      console.log("Claiming task as service role...");
      
      const { data, error } = await supabaseAdmin
        .from("tasks")
        .select("*")
        .eq("status", "Queued")
        .order("created_at", { ascending: true })
        .limit(1)
        .single();

      if (error && error.code !== "PGRST116") { // PGRST116 = no rows
        throw error;
      }

      if (data) {
        // Found a task - claim it atomically
        const { data: updateData, error: updateError } = await supabaseAdmin
          .from("tasks")
          .update({
            status: "In Progress",
            worker_id: workerId,
            updated_at: new Date().toISOString()
          })
          .eq("id", data.id)
          .eq("status", "Queued") // Prevent race conditions
          .select()
          .single();

        if (updateError || !updateData) {
          // Task was claimed by someone else, no task available
          rpcResponse = { data: [], error: null };
        } else {
          // Successfully claimed
          rpcResponse = {
            data: [{
              task_id_out: updateData.id,
              params_out: updateData.params,
              task_type_out: updateData.task_type,
              project_id_out: updateData.project_id
            }],
            error: null
          };
        }
      } else {
        // No tasks available
        rpcResponse = { data: [], error: null };
      }
    } else {
      // User token: use the user-specific claim function
      console.log(`Claiming task for user ${callerId}...`);
      
      try {
        // Try the user-specific function first
        // Query for queued tasks from projects owned by this user
        const { data, error } = await supabaseAdmin
          .from("tasks")
          .select(`
            *,
            project:projects!inner(user_id)
          `)
          .eq("status", "Queued")
          .eq("project.user_id", callerId)
          .order("created_at", { ascending: true })
          .limit(1)
          .single();

        if (error && error.code !== "PGRST116") { // PGRST116 = no rows
          throw error;
        }

        if (data) {
          // Found a task - claim it atomically
          const { data: updateData, error: updateError } = await supabaseAdmin
            .from("tasks")
            .update({
              status: "In Progress",
              worker_id: workerId,
              updated_at: new Date().toISOString()
            })
            .eq("id", data.id)
            .eq("status", "Queued") // Prevent race conditions
            .select()
            .single();

          if (updateError || !updateData) {
            // Task was claimed by someone else, no task available
            rpcResponse = { data: [], error: null };
          } else {
            // Successfully claimed
            rpcResponse = {
              data: [{
                task_id_out: updateData.id,
                params_out: updateData.params,
                task_type_out: updateData.task_type,
                project_id_out: updateData.project_id
              }],
              error: null
            };
          }
        } else {
          // No tasks available
          rpcResponse = { data: [], error: null };
        }
      } catch (e) {
        console.error("Error claiming task for user:", e);
        rpcResponse = { data: [], error: e };
      }
    }

    // Check RPC response
    if (rpcResponse.error) {
      console.error("RPC error:", rpcResponse.error);
      return new Response(`Database error: ${rpcResponse.error.message}`, { status: 500 });
    }

    // Check if we got a task
    if (!rpcResponse.data || rpcResponse.data.length === 0) {
      console.log("No queued tasks available");
      return new Response(null, { status: 204 });
    }

    const task = rpcResponse.data[0];
    console.log(`Successfully claimed task ${task.task_id_out}`);

    // Return the task data
    return new Response(JSON.stringify({
      task_id: task.task_id_out,
      params: task.params_out,
      task_type: task.task_type_out,
      project_id: task.project_id_out
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });

  } catch (error) {
    console.error("Unexpected error:", error);
    return new Response(`Internal server error: ${error.message}`, { status: 500 });
  }
}); 