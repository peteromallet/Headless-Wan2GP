-- ============================================================================
-- FIX WORKER FOREIGN KEY CONSTRAINT ISSUE
-- ============================================================================
-- This migration addresses the foreign key constraint violations where
-- worker_ids are referenced in tasks but don't exist in the workers table.
-- 
-- Solutions provided:
-- 1. Auto-insert missing workers when tasks reference them
-- 2. Create common worker entries for typical patterns
-- ============================================================================

-- ============================================================================
-- 1. CREATE TRIGGER to auto-insert workers when referenced in tasks
-- ============================================================================
CREATE OR REPLACE FUNCTION auto_create_worker()
RETURNS TRIGGER AS $$
BEGIN
    -- If worker_id is provided and doesn't exist in workers table, create it
    IF NEW.worker_id IS NOT NULL AND NEW.worker_id != '' THEN
        INSERT INTO workers (id, instance_type, status, last_heartbeat, metadata, created_at)
        VALUES (
            NEW.worker_id,
            CASE 
                WHEN NEW.worker_id LIKE 'edge_%' THEN 'edge_function'
                WHEN NEW.worker_id LIKE 'worker_%' THEN 'process'
                WHEN NEW.worker_id LIKE 'gpu-%' THEN 'gpu_worker'
                WHEN NEW.worker_id LIKE 'test_%' THEN 'test'
                ELSE 'external'
            END,
            'active',
            NOW(),
            jsonb_build_object(
                'auto_created', true,
                'created_by_trigger', true,
                'pattern_detected', CASE 
                    WHEN NEW.worker_id LIKE 'edge_%' THEN 'edge_function'
                    WHEN NEW.worker_id LIKE 'worker_%' THEN 'process'
                    WHEN NEW.worker_id LIKE 'gpu-%' THEN 'gpu_worker'
                    WHEN NEW.worker_id LIKE 'test_%' THEN 'test'
                    ELSE 'unknown'
                END
            ),
            NOW()
        )
        ON CONFLICT (id) DO NOTHING; -- Don't error if worker already exists
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on tasks table
DROP TRIGGER IF EXISTS trigger_auto_create_worker ON tasks;
CREATE TRIGGER trigger_auto_create_worker
    BEFORE INSERT OR UPDATE OF worker_id ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION auto_create_worker();

-- ============================================================================
-- 2. CREATE common worker entries for existing patterns
-- ============================================================================

-- Insert workers for any existing worker_ids in tasks table
INSERT INTO workers (id, instance_type, status, last_heartbeat, metadata, created_at)
SELECT DISTINCT 
    worker_id,
    CASE 
        WHEN worker_id LIKE 'edge_%' THEN 'edge_function'
        WHEN worker_id LIKE 'worker_%' THEN 'process'
        WHEN worker_id LIKE 'gpu-%' THEN 'gpu_worker'
        WHEN worker_id LIKE 'test_%' THEN 'test'
        ELSE 'external'
    END as instance_type,
    'active' as status,
    NOW() as last_heartbeat,
    jsonb_build_object(
        'auto_created', true,
        'backfilled_from_tasks', true,
        'source', 'existing_tasks'
    ) as metadata,
    NOW() as created_at
FROM tasks 
WHERE worker_id IS NOT NULL 
  AND worker_id != ''
  AND worker_id NOT IN (SELECT id FROM workers)
ON CONFLICT (id) DO NOTHING;

-- Create some common worker entries
INSERT INTO workers (id, instance_type, status, last_heartbeat, metadata, created_at) 
VALUES 
    ('default_worker', 'external', 'active', NOW(), '{"is_default": true}'::jsonb, NOW()),
    ('gpu-20250723_145828-38ab706b', 'gpu_worker', 'active', NOW(), '{"example_worker": true}'::jsonb, NOW()),
    ('gpu-20250723_221138-afa8403b', 'gpu_worker', 'active', NOW(), '{"specific_test_worker": true}'::jsonb, NOW())
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 3. VERIFICATION
-- ============================================================================
SELECT 'Workers auto-creation setup completed!' as result;

-- Show worker statistics
SELECT 
    instance_type, 
    COUNT(*) as count,
    STRING_AGG(id, ', ') as examples
FROM workers 
GROUP BY instance_type
ORDER BY instance_type; 