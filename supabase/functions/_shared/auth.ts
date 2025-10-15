/**
 * Shared authentication utilities for Supabase Edge Functions
 * 
 * Handles authentication for both:
 * - Service role keys (workers using service role)
 * - User tokens (PATs stored in user_api_tokens table)
 */

import { createClient, SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.7";

export interface AuthResult {
  success: boolean;
  isServiceRole: boolean;
  userId: string | null;
  error?: string;
  statusCode?: number;
}

/**
 * Authenticate a request using Bearer token
 * Supports both service role keys and user PATs
 */
export async function authenticateRequest(
  req: Request,
  supabaseAdmin: SupabaseClient,
  logPrefix: string = "[AUTH]"
): Promise<AuthResult> {
  // Extract authorization header
  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    console.log(`${logPrefix} Missing or invalid Authorization header`);
    return {
      success: false,
      isServiceRole: false,
      userId: null,
      error: "Missing or invalid Authorization header",
      statusCode: 401
    };
  }

  const token = authHeader.slice(7); // Remove "Bearer " prefix
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

  if (!serviceKey) {
    console.error(`${logPrefix} SUPABASE_SERVICE_ROLE_KEY not configured`);
    return {
      success: false,
      isServiceRole: false,
      userId: null,
      error: "Server configuration error",
      statusCode: 500
    };
  }

  let isServiceRole = false;
  let callerId: string | null = null;

  // 1) Check if token matches service-role key directly
  if (token === serviceKey) {
    isServiceRole = true;
    console.log(`${logPrefix} Direct service-role key match`);
    return {
      success: true,
      isServiceRole: true,
      userId: null
    };
  }

  // 2) If not service key, try to decode as JWT and check role
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
        console.log(`${logPrefix} JWT has service-role/admin role`);
        return {
          success: true,
          isServiceRole: true,
          userId: null
        };
      }
      // Don't extract user ID from JWT - always look it up in user_api_token table
    }
  } catch (e) {
    // Not a valid JWT - will be treated as PAT
    console.log(`${logPrefix} Token is not a valid JWT, treating as PAT`);
  }

  // 3) USER TOKEN PATH - resolve callerId via user_api_token table
  console.log(`${logPrefix} Looking up token in user_api_token table...`);
  
  try {
    const { data, error } = await supabaseAdmin
      .from("user_api_tokens")
      .select("user_id")
      .eq("token", token)
      .single();

    if (error || !data) {
      console.error(`${logPrefix} Token lookup failed:`, error);
      return {
        success: false,
        isServiceRole: false,
        userId: null,
        error: "Invalid or expired token",
        statusCode: 403
      };
    }

    callerId = data.user_id;
    console.log(`${logPrefix} Token resolved to user ID: ${callerId}`);
    return {
      success: true,
      isServiceRole: false,
      userId: callerId
    };
  } catch (e) {
    console.error(`${logPrefix} Error querying user_api_token:`, e);
    return {
      success: false,
      isServiceRole: false,
      userId: null,
      error: "Token validation failed",
      statusCode: 403
    };
  }
}

/**
 * Verify that a user owns a specific task
 * Used for user token authorization
 */
export async function verifyTaskOwnership(
  supabaseAdmin: SupabaseClient,
  taskId: string,
  userId: string,
  logPrefix: string = "[AUTH]"
): Promise<{ success: boolean; error?: string; statusCode?: number }> {
  try {
    const { data: taskData, error: taskError } = await supabaseAdmin
      .from("tasks")
      .select("user_id, project_id")
      .eq("id", taskId)
      .single();

    if (taskError || !taskData) {
      console.error(`${logPrefix} Task ${taskId} not found:`, taskError);
      return {
        success: false,
        error: "Task not found",
        statusCode: 404
      };
    }

    // Check if user owns the task directly
    if (taskData.user_id === userId) {
      console.log(`${logPrefix} User ${userId} owns task ${taskId} directly`);
      return { success: true };
    }

    // Check if user owns the project the task belongs to
    if (taskData.project_id) {
      const { data: projectData, error: projectError } = await supabaseAdmin
        .from("projects")
        .select("user_id")
        .eq("id", taskData.project_id)
        .single();

      if (!projectError && projectData && projectData.user_id === userId) {
        console.log(`${logPrefix} User ${userId} owns project ${taskData.project_id} for task ${taskId}`);
        return { success: true };
      }
    }

    console.log(`${logPrefix} User ${userId} does not own task ${taskId}`);
    return {
      success: false,
      error: "Forbidden: You do not own this task",
      statusCode: 403
    };
  } catch (e) {
    console.error(`${logPrefix} Error verifying task ownership:`, e);
    return {
      success: false,
      error: "Failed to verify task ownership",
      statusCode: 500
    };
  }
}

/**
 * Get the user_id that owns a specific task
 * Used by service role to determine storage path
 */
export async function getTaskUserId(
  supabaseAdmin: SupabaseClient,
  taskId: string,
  logPrefix: string = "[AUTH]"
): Promise<{ userId?: string; error?: string; statusCode?: number }> {
  try {
    const { data: taskData, error: taskError } = await supabaseAdmin
      .from("tasks")
      .select("user_id")
      .eq("id", taskId)
      .single();

    if (taskError || !taskData) {
      console.error(`${logPrefix} Task ${taskId} not found:`, taskError);
      return {
        error: "Task not found",
        statusCode: 404
      };
    }

    if (!taskData.user_id) {
      console.error(`${logPrefix} Task ${taskId} has no user_id`);
      return {
        error: "Task has no associated user",
        statusCode: 400
      };
    }

    return { userId: taskData.user_id };
  } catch (e) {
    console.error(`${logPrefix} Error getting task user:`, e);
    return {
      error: "Failed to get task user",
      statusCode: 500
    };
  }
}

