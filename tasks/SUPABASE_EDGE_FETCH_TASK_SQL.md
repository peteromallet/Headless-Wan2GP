# SQL Function for User-Filtered Task Claiming

Add this SQL function to your Supabase database to support the claim-next-task edge function:

```sql
-- Function to claim a task for a specific user (respects project_id filter)
CREATE OR REPLACE FUNCTION func_claim_user_task(
    p_table_name TEXT,
    p_worker_id TEXT,
    p_user_id TEXT
)
RETURNS TABLE(
    task_id_out TEXT,
    params_out JSONB,
    task_type_out TEXT,
    project_id_out TEXT
) AS $$
BEGIN
    RETURN QUERY EXECUTE format('
        WITH selected_task AS (
            SELECT t.id, t.task_id, t.params, t.task_type, t.project_id
            FROM %I t
            LEFT JOIN %I d ON d.id = t.dependant_on
            WHERE t.status = ''Queued''
              AND t.project_id = $2  -- Filter by user_id
              AND (t.dependant_on IS NULL OR d.status = ''Complete'')
            ORDER BY t.created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ), updated_task AS (
            UPDATE %I
            SET
                status = ''In Progress'',
                worker_id = $1,
                updated_at = CURRENT_TIMESTAMP,
                generation_started_at = CURRENT_TIMESTAMP
            WHERE id = (SELECT st.id FROM selected_task st)
            RETURNING task_id, params, task_type, project_id
        )
        SELECT ut.task_id, ut.params, ut.task_type, ut.project_id 
        FROM updated_task ut 
        LIMIT 1',
        p_table_name, p_table_name, p_table_name
    )
    USING p_worker_id, p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION func_claim_user_task TO authenticated;
GRANT EXECUTE ON FUNCTION func_claim_user_task TO service_role;
```

## Notes

1. This function is similar to `func_claim_task` but adds a `p_user_id` parameter
2. It filters tasks by `project_id = p_user_id` to ensure users only claim their own tasks
3. The `FOR UPDATE SKIP LOCKED` ensures atomic claiming in concurrent environments
4. The function checks dependencies (tasks with `dependant_on` only become available when the dependency is complete)
5. The `SECURITY DEFINER` means it runs with the privileges of the function owner (usually the database owner) 