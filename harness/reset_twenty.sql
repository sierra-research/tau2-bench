-- Hard reset for the Twenty CRM benchmark database.
-- SAFETY: never TRUNCATE ... CASCADE here. CASCADE follows foreign keys into
-- unrelated tables and can wipe workspace/user/auth rows on a shared instance.
--
-- Twenty stores CRM objects in a per-workspace schema such as
-- workspace_<id>, not in public. Discover every workspace schema that has a
-- person table and delete only the object rows there. The core schema
-- (workspace, user, workspaceMember, keyValue, metadata) is never touched.

DO $$
DECLARE
    s text;
    t text;
    object_tables text[] := ARRAY[
        'taskTarget', 'task_target',
        'noteTarget', 'note_target',
        'timelineActivity', 'timeline_activity',
        'favorite', 'attachment', 'note',
        'task',
        'opportunity',
        'person',
        'company'
    ];
    found boolean := false;
BEGIN
    FOR s IN
        SELECT n.nspname
        FROM pg_namespace n
        WHERE n.nspname LIKE 'workspace\_%'
          AND EXISTS (
              SELECT 1
              FROM pg_tables p
              WHERE p.schemaname = n.nspname
                AND p.tablename = 'person'
          )
        ORDER BY n.nspname
    LOOP
        found := true;
        RAISE NOTICE 'resetting workspace schema %', s;
        FOREACH t IN ARRAY object_tables LOOP
            IF to_regclass(format('%I.%I', s, t)) IS NOT NULL THEN
                EXECUTE format('DELETE FROM %I.%I', s, t);
            END IF;
        END LOOP;
    END LOOP;

    IF NOT found THEN
        RAISE EXCEPTION
            'No workspace_* schema containing a person table was found';
    END IF;
END $$;

-- Verify the CRM objects are empty in the workspace schema:
-- SELECT schemaname, tablename FROM pg_tables
--   WHERE schemaname LIKE 'workspace\_%' AND tablename IN
--   ('person','company','opportunity','task');
--
-- Verify login data is intact:
-- SELECT COUNT(*) FROM core.workspace;
-- SELECT COUNT(*) FROM core."user";
