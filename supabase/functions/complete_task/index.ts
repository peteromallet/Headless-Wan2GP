import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.7";

/**
 * Edge function:
 *   POST { task_id, file_data: "base64...", filename: "image.png" }
 *   Uploads file to Supabase Storage AND updates tasks row atomically
 *   -> sets status = 'Complete', output_location = public URL
 */
serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", {
      status: 405
    });
  }

  let body;
  try {
    body = await req.json();
  } catch (e) {
    return new Response("Invalid JSON body", {
      status: 400
    });
  }

  const { task_id, file_data, filename } = body;
  
  if (!task_id || !file_data || !filename) {
    return new Response("task_id, file_data (base64), and filename required", {
      status: 400
    });
  }

  // Admin client inside the function – uses service-role key injected by Supabase
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!, 
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  try {
    // 1. Decode the base64 file data
    const fileBuffer = Uint8Array.from(atob(file_data), c => c.charCodeAt(0));
    
    // 2. Get the authenticated user's ID from the request headers
    const authHeader = req.headers.get("Authorization");
    let userId = null;
    
    if (authHeader?.startsWith("Bearer ")) {
      const jwt = authHeader.substring(7);
      try {
        // Decode JWT to get user ID (simple base64 decode of payload)
        const [, payload] = jwt.split('.');
        const decodedPayload = JSON.parse(atob(payload + '=='.substring(0, (4 - payload.length % 4) % 4)));
        userId = decodedPayload.sub;
      } catch (e) {
        console.error("Failed to decode JWT:", e);
      }
    }

    // 3. Determine the storage path (with user ID prefix for RLS compliance)
    const objectPath = userId ? `${userId}/${filename}` : filename;
    
    // 4. Upload to Supabase Storage
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('image_uploads')
      .upload(objectPath, fileBuffer, {
        contentType: getContentType(filename),
        upsert: true
      });

    if (uploadError) {
      console.error("Storage upload error:", uploadError);
      return new Response(`Storage upload failed: ${uploadError.message}`, {
        status: 500
      });
    }

    // 5. Get the public URL
    const { data: urlData } = supabase.storage
      .from('image_uploads')
      .getPublicUrl(objectPath);
    
    const publicUrl = urlData.publicUrl;

    // 6. Update the database with the public URL
    const { error: dbError } = await supabase
      .from("tasks")
      .update({
        status: "Complete",
        output_location: publicUrl
      })
      .eq("id", task_id)
      .eq("status", "In Progress");

    if (dbError) {
      console.error("Database update error:", dbError);
      // If DB update fails, we should clean up the uploaded file
      await supabase.storage.from('image_uploads').remove([objectPath]);
      return new Response(`Database update failed: ${dbError.message}`, {
        status: 500
      });
    }

    return new Response(JSON.stringify({ 
      success: true, 
      public_url: publicUrl,
      message: "Task completed and file uploaded successfully" 
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });

  } catch (error) {
    console.error("Edge function error:", error);
    return new Response(`Internal error: ${error.message}`, {
      status: 500
    });
  }
});

function getContentType(filename: string): string {
  const ext = filename.toLowerCase().split('.').pop();
  switch (ext) {
    case 'png': return 'image/png';
    case 'jpg':
    case 'jpeg': return 'image/jpeg';
    case 'gif': return 'image/gif';
    case 'webp': return 'image/webp';
    case 'mp4': return 'video/mp4';
    case 'webm': return 'video/webm';
    case 'mov': return 'video/quicktime';
    default: return 'application/octet-stream';
  }
} 