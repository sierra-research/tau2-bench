        // Packet index: progress + flat-schema CSV export/import.
        // (rater/storage/CSV plumbing comes from form_core.js)

        // Realism dimension IDs (CSV columns for user_realism / voice_review).
        const DIM_IDS = (CFG.dims || []).map(d => d.id);

        // Set one annotation field from a header cell (import inverse of
        // form_core's annValueForHeader).
        function setAnnFieldFromHeader(base, header, value) {
            switch (header) {
                case 'batch': base.batch = value || BATCH_NAME; return;
                case 'rater': base.rater = value || null; return;
                case 'task_id': base.task_id = value; return;
                case 'simulation_id': base.simulation_id = value; return;
                case 'trial': base.trial = parseInt(value) || 0; return;
                case 'error_source': base.summary_error_source = value || null; return;
                case 'error_type': base.summary_error_type = value || null; return;
                case 'notes': base.summary_notes = value || ''; return;
                case 'free_text_comments': base.free_text_comments = value || ''; return;
                case 'caller_experience': base.caller_experience = value ? parseInt(value) : null; return;
                case 'experience_breaking_point': base.experience_breaking_point = value || null; return;
                case 'experience_breaking_tick': base.experience_breaking_tick = value ? parseInt(value) : null; return;
                case 'experience_factors':
                    base.experience_factors = (value || '').split(/[,;]/).map(s => s.trim()).filter(Boolean);
                    return;
                case 'primary_factor': base.primary_factor = value || null; return;
                case 'experience_notes': base.experience_notes = value || ''; return;
                case 'completed': base.completed = value === 'true' || value === 'True' || value === '1'; return;
                case 'created_at': base.created_at = value || new Date().toISOString(); return;
            }
            if (header.endsWith('_notes') && DIM_IDS.includes(header.slice(0, -6))) {
                base[`dim_${header.slice(0, -6)}_notes`] = value || '';
                return;
            }
            if (DIM_IDS.includes(header)) {
                const v = (value ?? '').trim();
                // Same NA variants norm_likert_na absorbs on the python side.
                base[`dim_${header}`] = v === ''
                    ? null
                    : (['NA', 'N/A'].includes(v.toUpperCase()) ? 'NA' : parseInt(v));
            }
        }

        function csvRowToAnnotation(row, headers) {
            const base = {
                id: `imported_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                task_type: CFG.form,
            };
            headers.forEach((h, idx) => { setAnnFieldFromHeader(base, h, dearmorCSVField(row[idx])); });
            return base;
        }

        function exportToCSV() {
            const stored = getStoredAnnotations();
            const annotations = Object.values(stored);

            if (annotations.length === 0) {
                alert('No annotations saved yet!');
                return;
            }

            downloadCSVBackup(flatCSVText(annotations));
        }

        function onAnnotationsCleared() {
            updateStatusIndicators();
        }

        function importFromCSV(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const csv = e.target.result;
                    const rows = parseCSV(csv);

                    if (rows.length < 2) {
                        showImportStatus('CSV file is empty or has no data rows', true);
                        return;
                    }

                    const headers = rows[0];
                    const requiredFields = ['task_id', 'simulation_id'];
                    for (const field of requiredFields) {
                        if (!headers.includes(field)) {
                            showImportStatus(`Missing required column: ${field}`, true);
                            return;
                        }
                    }

                    const stored = getStoredAnnotations();
                    let importCount = 0;
                    let skippedCount = 0;

                    for (let i = 1; i < rows.length; i++) {
                        const values = rows[i];
                        if (values.length !== headers.length) { skippedCount++; continue; }

                        const key = `${dearmorCSVField(values[headers.indexOf('task_id')])}_${dearmorCSVField(values[headers.indexOf('simulation_id')])}`;
                        stored[key] = csvRowToAnnotation(values, headers);
                        importCount++;
                    }

                    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
                    updateStatusIndicators();
                    if (skippedCount > 0) {
                        showImportStatus(
                            `Imported ${importCount} annotations; skipped ${skippedCount} malformed row(s) (wrong column count)`,
                            true
                        );
                    } else {
                        showImportStatus(`Successfully imported ${importCount} annotations`, false);
                    }

                } catch (err) {
                    showImportStatus('Error parsing CSV: ' + err.message, true);
                }
            };
            reader.readAsText(file);
            event.target.value = '';
        }

        // Column sorting
        wireIndexSorting();

        // Update status on page load
        updateStatusIndicators();
