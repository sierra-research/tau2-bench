        // Shared packet-form plumbing: rater identity + rater-scoped storage
        // keys, localStorage annotation persistence, CSV escape/parse, backup
        // downloads, status toasts, and the index progress/sorting helpers.
        // Inlined before every form/index script; per-form divergence
        // (storage schema, CSV shape, completion gates) comes in through
        // parameters and page-defined hooks (initializeForm,
        // generateAnnotation, onAnnotationsCleared — function declarations
        // hoisted from the same inlined script block), never as redefined
        // copies.
        //
        // FROZEN CONTRACT: the localStorage keys
        // (tau2_ann_<batch_id>_rater, tau2_ann_<batch_id>_annotations[_<rater>])
        // and every CSV shape produced here must stay byte-compatible with
        // in-flight annotator packets.
        const CFG = window.PACKET_CONFIG;
        const BATCH_NAME = CFG.batch_name;
        const RATER_KEY = `tau2_ann_${CFG.batch_id}_rater`;

        // Sim/task identity (index pages carry no sim; these stay unused there).
        const annotationMeta = {
            simulation_id: CFG.sim_id,
            task_id: CFG.task_id,
            trial: CFG.trial
        };
        // Pair pages (preference_pair) key storage by pair_id; sim pages by
        // task+sim. Same frozen-key rules apply to both.
        const ANNOTATION_KEY = CFG.pair_id ? CFG.pair_id : `${CFG.task_id}_${CFG.sim_id}`;

        function getRaterName() {
            return localStorage.getItem(RATER_KEY);
        }

        function promptRaterName() {
            const overlay = document.getElementById('raterModal');
            overlay.classList.add('active');
            document.getElementById('raterNameInput').focus();
        }

        function submitRaterName() {
            const name = document.getElementById('raterNameInput').value.trim();
            if (!name) return;
            localStorage.setItem(RATER_KEY, name);
            document.getElementById('raterModal').classList.remove('active');
            document.getElementById('raterDisplay').textContent = name;
            initPage();
        }

        let RATER_NAME = getRaterName();
        const STORAGE_KEY = RATER_NAME
            ? `tau2_ann_${CFG.batch_id}_annotations_${RATER_NAME}`
            : `tau2_ann_${CFG.batch_id}_annotations`;

        // After a rater-name change: a changed storage key forces a reload;
        // otherwise refresh the badge and re-run the page's restore hook
        // (form pages define initializeForm()).
        function initPage() {
            RATER_NAME = getRaterName();
            const newKey = `tau2_ann_${CFG.batch_id}_annotations_${RATER_NAME}`;
            if (newKey !== STORAGE_KEY) {
                window.location.reload();
                return;
            }
            const badge = document.getElementById('raterDisplay');
            if (badge) badge.textContent = RATER_NAME;
            if (typeof initializeForm === 'function') initializeForm();
        }

        if (!RATER_NAME) {
            document.addEventListener('DOMContentLoaded', promptRaterName);
        } else {
            document.addEventListener('DOMContentLoaded', () => {
                const badge = document.getElementById('raterDisplay');
                if (badge) {
                    badge.textContent = RATER_NAME;
                    badge.title = 'Click to change name';
                    badge.style.cursor = 'pointer';
                    badge.addEventListener('click', promptRaterName);
                }
            });
        }

        function getStoredAnnotations() {
            try {
                return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
            } catch (e) {
                return {};
            }
        }

        // Persist the form's current annotation under this sim's key.
        // generateAnnotation() is the form page's serializer.
        function saveToLocalStorage() {
            const annotation = generateAnnotation();
            const stored = getStoredAnnotations();
            stored[ANNOTATION_KEY] = annotation;
            localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
            return annotation;
        }

        function showStatus(message, isError = false) {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = 'status-message ' + (isError ? 'error' : 'success');
            setTimeout(() => {
                statusEl.className = 'status-message';
            }, 3000);
        }

        function showImportStatus(message, isError) {
            const el = document.getElementById('importStatus');
            el.textContent = message;
            el.className = 'import-status ' + (isError ? 'error' : 'success');
            setTimeout(() => { el.className = 'import-status'; }, 5000);
        }

        function escapeCSV(val) {
            if (val === null || val === undefined) return '';
            val = String(val);
            // Formula-injection armor: spreadsheet apps execute fields that
            // start with = + - @ (or tab/CR). A leading apostrophe keeps
            // them inert text; import strips it back off.
            if (/^[=+\-@\t\r]/.test(val)) {
                val = "'" + val;
            }
            if (val.includes(',') || val.includes('\n') || val.includes('\r') || val.includes('"')) {
                val = '"' + val.replace(/"/g, '""') + '"';
            }
            return val;
        }

        const FORMULA_GUARD_PREFIXES = ['=', '+', '-', '@', '\t', '\r'];

        // Inverse of the escapeCSV formula guard: strip ONE armoring apostrophe,
        // but preserve a genuine leading apostrophe in annotator-authored text.
        function dearmorCSVField(val) {
            return (typeof val === 'string' && val.startsWith("'") &&
                FORMULA_GUARD_PREFIXES.includes(val[1])) ? val.slice(1) : val;
        }

        function parseCSV(text) {
            const rows = [];
            let current = [];
            let field = '';
            let inQuotes = false;

            for (let i = 0; i < text.length; i++) {
                const ch = text[i];

                if (inQuotes) {
                    if (ch === '"' && text[i + 1] === '"') {
                        field += '"';
                        i++;
                    } else if (ch === '"') {
                        inQuotes = false;
                    } else {
                        field += ch;
                    }
                } else if (ch === '"') {
                    inQuotes = true;
                } else if (ch === ',') {
                    current.push(field);
                    field = '';
                } else if (ch === '\r') {
                    // skip \r
                } else if (ch === '\n') {
                    current.push(field);
                    field = '';
                    if (current.some(f => f.trim())) rows.push(current);
                    current = [];
                } else {
                    field += ch;
                }
            }
            current.push(field);
            if (current.some(f => f.trim())) rows.push(current);
            return rows;
        }

        // Map one PACKET_CONFIG.csv_headers header to its annotation value.
        // Superset of every flat-schema form's headers — a form's CSV only
        // ever asks for the headers its row model declares.
        function annValueForHeader(ann, header) {
            switch (header) {
                case 'batch': return ann.batch || BATCH_NAME;
                case 'rater': return ann.rater || RATER_NAME;
                case 'task_id': return ann.task_id;
                case 'simulation_id': return ann.simulation_id;
                case 'trial': return ann.trial;
                case 'error_source': return ann.summary_error_source || '';
                case 'error_type': return ann.summary_error_type || '';
                case 'notes': return ann.summary_notes || '';
                case 'free_text_comments': return ann.free_text_comments || '';
                case 'caller_experience': return ann.caller_experience ?? '';
                case 'experience_breaking_point': return ann.experience_breaking_point || '';
                case 'experience_breaking_tick': return ann.experience_breaking_tick ?? '';
                case 'experience_factors': return (ann.experience_factors || []).join(', ');
                case 'primary_factor': return ann.primary_factor || '';
                case 'experience_notes': return ann.experience_notes || '';
                case 'completed': return ann.completed;
                case 'created_at': return ann.created_at;
            }
            if (header.endsWith('_notes')) return ann[`dim_${header.slice(0, -6)}_notes`] ?? '';
            return ann[`dim_${header}`] ?? '';
        }

        function downloadFile(content, mime, filename) {
            const blob = new Blob([content], { type: mime });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // The rater-stamped CSV download every backup/export path uses.
        function downloadCSVBackup(csv) {
            downloadFile(csv, 'text/csv',
                `${BATCH_NAME}_${RATER_NAME}_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.csv`);
        }

        // Flat-schema CSV text: one row per stored annotation.
        function flatCSVText(annotations) {
            const headers = CFG.csv_headers;
            let csv = headers.join(',') + '\n';
            for (const ann of annotations) {
                csv += headers.map(h => escapeCSV(annValueForHeader(ann, h))).join(',') + '\n';
            }
            return csv;
        }

        // Backup download on mark-complete (flat-schema forms; the VE form
        // downloads its long-format CSV through downloadCSVBackup directly).
        function autoBackupCSV() {
            const annotations = Object.values(getStoredAnnotations());
            if (annotations.length === 0) return;
            downloadCSVBackup(flatCSVText(annotations));
        }

        // Copy / download-JSON buttons on form pages. `save` persists the
        // current annotation and returns the object to serialize.
        function wireAnnotationButtons(save) {
            document.getElementById('copyAnnotationBtn').addEventListener('click', () => {
                const json = JSON.stringify(save(), null, 2);
                navigator.clipboard.writeText(json).then(() => {
                    showStatus('Saved & copied to clipboard!');
                }).catch(err => {
                    showStatus('Failed to copy: ' + err, true);
                });
            });
            document.getElementById('downloadAnnotationBtn').addEventListener('click', () => {
                const json = JSON.stringify(save(), null, 2);
                downloadFile(json, 'application/json',
                    `${BATCH_NAME}_${RATER_NAME}_task${annotationMeta.task_id}.json`);
                showStatus('Saved & downloaded!');
            });
        }

        // ------------------------------------------------------------------
        // Navigate-away flush (the static prev/next bars + back link). Text
        // fields normally persist on blur, which fires before a link click
        // lands — but flush explicitly on nav clicks and beforeunload anyway
        // so a typed-but-unblurred edit can never be lost to navigation.
        //
        // Two form shapes, ONE seam: forms whose draft lives in a state
        // object (preference_pair) define syncPendingEdits() —
        // hoisted from the same inlined script block — to pull unblurred DOM
        // text into that state and persist it; flat-schema forms serialize
        // straight from the live DOM (generateAnnotation), so a plain
        // saveToLocalStorage() already captures every pending edit.
        // ------------------------------------------------------------------

        let formTouched = false;
        ['input', 'change'].forEach(type => {
            document.addEventListener(type, (e) => {
                if (e.target instanceof Element && e.target.closest('#annotationForm')) {
                    formTouched = true;
                }
            }, true);
        });

        function flushPendingEdits() {
            // Never materialize a draft for a page the rater only LOOKED at:
            // an untouched form must not flip its index row to "in progress".
            if (!formTouched && !(ANNOTATION_KEY in getStoredAnnotations())) return;
            if (typeof syncPendingEdits === 'function') {
                syncPendingEdits();
            } else if (typeof generateAnnotation === 'function') {
                saveToLocalStorage();
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            document.querySelectorAll('.packet-nav a, .back-link').forEach(link => {
                link.addEventListener('click', flushPendingEdits);
            });
            window.addEventListener('beforeunload', flushPendingEdits);
        });

        // ------------------------------------------------------------------
        // Index-page plumbing (called only from the index scripts)
        // ------------------------------------------------------------------

        function updateStatusIndicators() {
            const stored = getStoredAnnotations();
            let completedCount = 0;
            let inProgressCount = 0;
            let pendingCount = 0;

            document.querySelectorAll('#taskList tr').forEach(li => {
                const taskId = li.dataset.task;
                const simId = li.dataset.sim;
                const statusEl = li.querySelector('.status');

                // Exact key match: task_id + "_" + simulation_id
                const key = `${taskId}_${simId}`;
                const annotation = stored[key];

                if (annotation && annotation.completed) {
                    statusEl.className = 'status done';
                    statusEl.textContent = 'done';
                    completedCount++;
                } else if (annotation) {
                    statusEl.className = 'status in-progress';
                    statusEl.textContent = 'in progress';
                    inProgressCount++;
                } else {
                    statusEl.className = 'status pending';
                    statusEl.textContent = 'pending';
                    pendingCount++;
                }
            });

            document.getElementById('completedCount').textContent = completedCount;
            document.getElementById('inProgressCount').textContent = inProgressCount;
            document.getElementById('pendingCount').textContent = pendingCount;
        }

        // Clear-all button; onAnnotationsCleared() is the index page's
        // refresh hook.
        function clearAnnotations() {
            if (confirm('Are you sure you want to clear all saved annotations? This cannot be undone.')) {
                localStorage.removeItem(STORAGE_KEY);
                onAnnotationsCleared();
                showImportStatus('All annotations cleared.', false);
            }
        }

        // Sortable index columns (task id / experiment / status).
        function wireIndexSorting() {
            document.querySelectorAll('th.sortable').forEach(th => {
                th.addEventListener('click', () => {
                    const sortKey = th.dataset.sort;
                    const isNumeric = th.dataset.type === 'number';
                    const tbody = document.getElementById('taskList');
                    const rows = Array.from(tbody.querySelectorAll('tr'));

                    const wasAsc = th.classList.contains('sort-asc');
                    document.querySelectorAll('th.sortable').forEach(h => {
                        h.classList.remove('sort-asc', 'sort-desc');
                    });
                    const dir = wasAsc ? 'desc' : 'asc';
                    th.classList.add(`sort-${dir}`);

                    rows.sort((a, b) => {
                        let va, vb;
                        if (sortKey === 'task') {
                            va = a.dataset.task || '';
                            vb = b.dataset.task || '';
                        } else if (sortKey === 'experiment') {
                            va = a.querySelector('.experiment-col')?.textContent || '';
                            vb = b.querySelector('.experiment-col')?.textContent || '';
                        } else if (sortKey === 'status') {
                            va = a.querySelector('.status')?.textContent || '';
                            vb = b.querySelector('.status')?.textContent || '';
                        }
                        let cmp;
                        if (isNumeric) {
                            const na = parseFloat(va);
                            const nb = parseFloat(vb);
                            // Non-numeric task ids fall back to string compare
                            // instead of collapsing to 0 (which froze the sort).
                            cmp = (!isNaN(na) && !isNaN(nb))
                                ? na - nb
                                : String(va).localeCompare(String(vb));
                        } else {
                            cmp = String(va).localeCompare(String(vb));
                        }
                        return dir === 'desc' ? -cmp : cmp;
                    });

                    rows.forEach(r => tbody.appendChild(r));
                });
            });
        }
