# Complete Task Edge Function

This Edge Function handles task completion by uploading files and updating task status.

## Memory Optimization

This function supports **three modes** to handle files of any size efficiently:

### Mode 1: Legacy JSON with Base64 (Backward Compatible)

**Use for:** Small files (<10MB), existing integrations

```javascript
const response = await fetch(`${supabaseUrl}/functions/v1/complete-task`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    task_id: 'uuid-here',
    file_data: base64EncodedString,
    filename: 'output.mp4',
    first_frame_data: base64ThumbnailString,  // optional
    first_frame_filename: 'thumbnail.jpg'      // optional
  })
});
```

**Memory Impact:** For a 100MB video, this requires ~233MB peak memory (133MB base64 + 100MB decoded).

### Mode 2: Streaming Multipart Upload (Recommended for Large Files)

**Use for:** Large files (>10MB), videos, high-resolution images

```javascript
const formData = new FormData();
formData.append('task_id', 'uuid-here');
formData.append('file', fileBlob, 'output.mp4');
formData.append('first_frame', thumbnailBlob, 'thumbnail.jpg'); // optional

const response = await fetch(`${supabaseUrl}/functions/v1/complete-task`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
    // Don't set Content-Type - let browser set it with boundary
  },
  body: formData
});
```

**Memory Impact:** Reduced by ~33% (no base64 inflation). Uses streaming multipart parser with progress logging. File is still buffered once for upload to Supabase Storage, but avoids double-buffering from base64 encoding.

### Mode 3: Pre-Signed URL Upload (**Recommended for Large Files >100MB**)

**Use for:** Large files (>100MB), videos that hit memory limits

This mode completely bypasses Edge Function memory limits by having the worker upload directly to storage.

**Step 1: Get signed upload URL**
```javascript
const response = await fetch(`${supabaseUrl}/functions/v1/generate-upload-url`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    task_id: 'uuid-here',
    filename: 'output.mp4',
    content_type: 'video/mp4',
    generate_thumbnail_url: true  // optional
  })
});

const { upload_url, storage_path, thumbnail_upload_url, thumbnail_storage_path } = await response.json();
```

**Step 2: Upload directly to storage**
```javascript
// Upload main file
await fetch(upload_url, {
  method: 'PUT',
  headers: { 'Content-Type': 'video/mp4' },
  body: fileBlob
});

// Upload thumbnail (optional)
if (thumbnail_upload_url) {
  await fetch(thumbnail_upload_url, {
    method: 'PUT',
    headers: { 'Content-Type': 'image/jpeg' },
    body: thumbnailBlob
  });
}
```

**Step 3: Complete task**
```javascript
await fetch(`${supabaseUrl}/functions/v1/complete-task`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    task_id: 'uuid-here',
    storage_path: storage_path,
    thumbnail_storage_path: thumbnail_storage_path  // optional
  })
});
```

**Memory Impact:** **Zero** Edge Function memory usage for file data. File goes directly to storage. Works with files of **any size** - 100MB, 500MB, 5GB+.

**Storage Path Format:** Files are organized by task: `userId/tasks/{task_id}/filename`. This provides:
- ✅ **Security**: Path validation ensures files match the task_id
- ✅ **Organization**: Easy to find all files for a specific task  
- ✅ **Cleanup**: Can delete entire task folder when task is removed

**Orchestrator Task Exception:** Orchestrator tasks (e.g., `travel_orchestrator`) can reference outputs from other tasks (typically child tasks like stitch/segment tasks) by passing their storage paths. Security is maintained through:
- Task ownership verification (caller must own the orchestrator task being completed)
- File existence validation (referenced file must exist in storage)
- Storage RLS policies (cross-user access already protected)

## Python Examples (for worker scripts)

### Mode 1: Legacy (Base64) - Works with existing code
```python
import requests
import base64

url = f"{supabase_url}/functions/v1/complete-task"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

with open("output.mp4", "rb") as f:
    file_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(url, headers=headers, json={
    "task_id": task_id,
    "file_data": file_data,
    "filename": "output.mp4"
})
```

### Mode 2: Streaming (Multipart) - Better for medium files
```python
import requests

url = f"{supabase_url}/functions/v1/complete-task"
headers = {"Authorization": f"Bearer {token}"}

with open("output.mp4", "rb") as video_file:
    files = {'file': ('output.mp4', video_file, 'video/mp4')}
    data = {'task_id': task_id}
    
    # Optional: Add thumbnail
    # with open("thumbnail.jpg", "rb") as thumb_file:
    #     files['first_frame'] = ('thumbnail.jpg', thumb_file, 'image/jpeg')
    
    response = requests.post(url, headers=headers, files=files, data=data)
```

### Mode 3: Pre-Signed URL - **Best for large files (>100MB)**
```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Step 1: Get signed upload URLs
gen_url_response = requests.post(
    f"{supabase_url}/functions/v1/generate-upload-url",
    headers=headers,
    json={
        "task_id": task_id,
        "filename": "output.mp4",
        "content_type": "video/mp4",
        "generate_thumbnail_url": True  # optional
    }
)
upload_data = gen_url_response.json()

# Step 2: Upload files directly to storage (no edge function memory!)
with open("output.mp4", "rb") as f:
    requests.put(
        upload_data["upload_url"],
        headers={"Content-Type": "video/mp4"},
        data=f
    )

# Optional: Upload thumbnail
if "thumbnail_upload_url" in upload_data:
    with open("thumbnail.jpg", "rb") as f:
        requests.put(
            upload_data["thumbnail_upload_url"],
            headers={"Content-Type": "image/jpeg"},
            data=f
        )

# Step 3: Complete task (just pass storage paths, no file data)
requests.post(
    f"{supabase_url}/functions/v1/complete-task",
    headers=headers,
    json={
        "task_id": task_id,
        "storage_path": upload_data["storage_path"],
        "thumbnail_storage_path": upload_data.get("thumbnail_storage_path")
    }
)
```

## Response Format

Both modes return the same JSON response:

```json
{
  "success": true,
  "public_url": "https://...",
  "thumbnail_url": "https://...",
  "message": "Task completed and file uploaded successfully"
}
```

## Error Responses

- `400` - Missing required fields or invalid data
- `401` - Missing or invalid Authorization header
- `403` - User not authorized for this task
- `404` - Task or project not found
- `500` - Internal server error

## Migration Guide

To migrate existing code to streaming mode:

1. Change `Content-Type` from `application/json` to multipart (or let FormData set it)
2. Replace base64 encoding with direct file upload via FormData
3. Replace `file_data`/`filename` with a single `file` field
4. Replace `first_frame_data`/`first_frame_filename` with a single `first_frame` field

### Before (Legacy):
```javascript
const base64 = await fileToBase64(file);
body: JSON.stringify({ task_id, file_data: base64, filename: file.name })
```

### After (Streaming):
```javascript
const formData = new FormData();
formData.append('task_id', task_id);
formData.append('file', file);
```

## Performance Comparison

| File Size | Mode 1 (Base64) | Mode 2 (Multipart) | Mode 3 (Pre-Signed URL) |
|-----------|----------------|-------------------|------------------------|
| 50MB      | ~116MB<br>(66MB base64 + 50MB) | ~55MB<br>(file only) | **~5MB**<br>(auth only) ✅ |
| 100MB     | ~233MB<br>(133MB base64 + 100MB) | ~105MB<br>(file only) | **~5MB**<br>(auth only) ✅ |
| 200MB     | ~466MB<br>**❌ Often fails** | ~205MB | **~5MB**<br>(auth only) ✅ |
| 500MB     | **❌ Memory Error** | ~505MB<br>**❌ May fail** | **~5MB**<br>(auth only) ✅ |
| 2GB       | **❌ Memory Error** | **❌ Memory Error** | **~5MB**<br>(auth only) ✅ |

**Recommendation by file size:**
- **<10MB:** Mode 1 (existing code works)
- **10-100MB:** Mode 2 (better performance, no code changes if using multipart)
- **>100MB:** **Mode 3** (zero memory usage, works with any file size)

**Mode 3 advantages:**
- ✅ Zero Edge Function memory usage for files
- ✅ Works with files of **any size** (100MB, 500MB, 5GB+)
- ✅ Direct upload to storage (faster)
- ✅ No timeout issues
- ✅ Secure (signed URLs expire in 1 hour, validate ownership)

## Troubleshooting

**"Memory limit exceeded" error:**
- Switch to streaming mode (multipart/form-data)
- This typically happens with videos >100MB in legacy JSON mode

**"Failed to parse form data" error:**
- Ensure you're not setting `Content-Type` header manually with FormData
- Browser/client should set it automatically with multipart boundary

**"Invalid base64 file_data" error:**
- Only in legacy JSON mode
- Check that base64 encoding is correct and not corrupted
