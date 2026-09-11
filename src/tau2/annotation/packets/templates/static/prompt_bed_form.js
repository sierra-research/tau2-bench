        // Prompt + bed review form: one page, one long-format CSV.
        // Storage/export conventions come from form_core.js (rater-scoped
        // localStorage keys namespaced by the content-derived batch id;
        // formula-armored CSV).

        // State: { items: { item_id: {verdict, notes, created_at} }, completed }
        function loadState() {
            try {
                const s = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
                return { items: s.items || {}, completed: !!s.completed };
            } catch (e) {
                return { items: {}, completed: false };
            }
        }

        let STATE = loadState();

        function saveState() {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(STATE));
            updateProgress();
        }

        function itemState(itemId) {
            if (!STATE.items[itemId]) {
                STATE.items[itemId] = { verdict: null, notes: '', created_at: '' };
            }
            return STATE.items[itemId];
        }

        function updateProgress() {
            const total = CFG.items.length;
            const reviewed = CFG.items.filter(it => {
                const s = STATE.items[it.item_id];
                return s && s.verdict;
            }).length;
            document.getElementById('reviewedCount').textContent = reviewed;
            document.getElementById('totalCount').textContent = total;
        }

        function refreshRow(row) {
            const itemId = row.dataset.item;
            const s = STATE.items[itemId] || {};
            row.querySelectorAll(`input[name="verdict_${itemId}"]`).forEach(r => {
                r.checked = !!s.verdict && r.value === s.verdict;
            });
            const notes = row.querySelector('.pb-notes');
            if (notes) notes.value = s.notes || '';
            row.classList.toggle('pb-has-issue', s.verdict === 'issue');
        }

        function wireRows() {
            document.querySelectorAll('.pb-verdict-row[data-item]').forEach(row => {
                const itemId = row.dataset.item;
                refreshRow(row);
                row.querySelectorAll(`input[name="verdict_${itemId}"]`).forEach(radio => {
                    radio.addEventListener('change', () => {
                        const s = itemState(itemId);
                        s.verdict = radio.value;
                        s.created_at = new Date().toISOString();
                        row.classList.toggle('pb-has-issue', s.verdict === 'issue');
                        saveState();
                    });
                });
                const notes = row.querySelector('.pb-notes');
                if (notes) {
                    notes.addEventListener('input', () => {
                        const s = itemState(itemId);
                        s.notes = notes.value;
                        s.created_at = new Date().toISOString();
                        saveState();
                    });
                }
            });
            const complete = document.getElementById('completeCheckbox');
            complete.checked = STATE.completed;
            complete.addEventListener('change', () => {
                STATE.completed = complete.checked;
                saveState();
            });
        }

        function refreshAll() {
            document.querySelectorAll('.pb-verdict-row[data-item]').forEach(refreshRow);
            document.getElementById('completeCheckbox').checked = STATE.completed;
            updateProgress();
        }

        // Map one PACKET_CONFIG.csv_headers header to an item's cell value.
        function cellForHeader(item, state, header) {
            switch (header) {
                case 'row_kind': return item.row_kind;
                case 'batch': return BATCH_NAME;
                case 'rater': return RATER_NAME || '';
                case 'language': return CFG.language;
                case 'item_id': return item.item_id;
                case 'locale': return item.locale;
                case 'bed_type': return item.bed_type;
                case 'verdict': return state.verdict || '';
                case 'notes': return state.notes || '';
                case 'completed': return STATE.completed;
                case 'created_at': return state.created_at || '';
            }
            return '';
        }

        function exportToCSV() {
            // EVERY item is exported, reviewed or not — a blank verdict means
            // "not reviewed yet" and must stay visible at ingest.
            const headers = CFG.csv_headers;
            let csv = headers.join(',') + '\n';
            for (const item of CFG.items) {
                const state = STATE.items[item.item_id] || {};
                csv += headers.map(h => escapeCSV(cellForHeader(item, state, h))).join(',') + '\n';
            }
            downloadCSVBackup(csv);
        }

        function onAnnotationsCleared() {
            STATE = loadState();
            refreshAll();
        }

        function importFromCSV(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const rows = parseCSV(e.target.result);
                    if (rows.length < 2) {
                        showImportStatus('CSV file is empty or has no data rows', true);
                        return;
                    }
                    const headers = rows[0];
                    for (const field of ['item_id', 'verdict', 'notes']) {
                        if (!headers.includes(field)) {
                            showImportStatus(`Missing required column: ${field}`, true);
                            return;
                        }
                    }
                    const idx = h => headers.indexOf(h);
                    const knownIds = new Set(CFG.items.map(it => it.item_id));
                    let importCount = 0;
                    let skippedCount = 0;
                    for (let i = 1; i < rows.length; i++) {
                        const values = rows[i];
                        if (values.length !== headers.length) { skippedCount++; continue; }
                        const itemId = dearmorCSVField(values[idx('item_id')]);
                        if (!knownIds.has(itemId)) { skippedCount++; continue; }
                        const verdict = dearmorCSVField(values[idx('verdict')]).toLowerCase();
                        const s = itemState(itemId);
                        s.verdict = (verdict === 'ok' || verdict === 'issue') ? verdict : null;
                        s.notes = dearmorCSVField(values[idx('notes')]) || '';
                        if (idx('created_at') >= 0) {
                            s.created_at = dearmorCSVField(values[idx('created_at')]) || '';
                        }
                        if (idx('completed') >= 0) {
                            const c = dearmorCSVField(values[idx('completed')]);
                            STATE.completed = c === 'true' || c === 'True' || c === '1';
                        }
                        importCount++;
                    }
                    saveState();
                    refreshAll();
                    if (skippedCount > 0) {
                        showImportStatus(
                            `Imported ${importCount} rows; skipped ${skippedCount} malformed or unknown row(s)`,
                            true
                        );
                    } else {
                        showImportStatus(`Successfully imported ${importCount} rows`, false);
                    }
                } catch (err) {
                    showImportStatus('Error parsing CSV: ' + err.message, true);
                }
            };
            reader.readAsText(file);
            event.target.value = '';
        }

        document.addEventListener('DOMContentLoaded', () => {
            wireRows();
            updateProgress();
        });
