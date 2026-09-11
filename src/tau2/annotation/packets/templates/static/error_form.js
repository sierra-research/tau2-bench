        // Annotation form handling - Error Editor
        // (rater/storage/CSV plumbing comes from form_core.js)

        function initializeForm() {
            const stored = getStoredAnnotations();
            const saved = stored[ANNOTATION_KEY];

            if (saved) {
                document.getElementById('summaryErrorSource').value = saved.summary_error_source || '';
                document.getElementById('summaryErrorType').value = saved.summary_error_type || '';
                document.getElementById('summaryNotes').value = saved.summary_notes || '';
                document.getElementById('markComplete').checked = saved.completed || false;
                showStatus(saved.completed ? 'Loaded completed annotation' : 'Loaded saved annotation');
            }
        }

        function generateAnnotation() {
            return {
                id: `annotation_${Date.now().toString(36)}`,
                simulation_id: annotationMeta.simulation_id,
                task_id: annotationMeta.task_id,
                trial: annotationMeta.trial,
                summary_error_source: document.getElementById('summaryErrorSource').value || null,
                summary_error_type: document.getElementById('summaryErrorType').value || null,
                summary_notes: document.getElementById('summaryNotes').value,
                rater: RATER_NAME,
                batch: BATCH_NAME,
                created_at: new Date().toISOString(),
                completed: document.getElementById('markComplete').checked,
            };
        }

        // Initialize on page load
        initializeForm();

        // Form fields auto-save
        ['summaryErrorSource', 'summaryErrorType', 'summaryNotes'].forEach(id => {
            const el = document.getElementById(id);
            el.addEventListener('change', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
            if (el.tagName === 'TEXTAREA') {
                el.addEventListener('blur', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
            }
        });

        // Mark complete checkbox
        document.getElementById('markComplete').addEventListener('change', () => {
            const isComplete = document.getElementById('markComplete').checked;
            saveToLocalStorage();
            if (isComplete) {
                autoBackupCSV();
                showStatus('Marked as complete! Backup CSV downloaded.');
            } else {
                showStatus('Marked as in progress');
            }
        });

        wireAnnotationButtons(saveToLocalStorage);
