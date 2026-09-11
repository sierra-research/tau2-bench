        // Realism annotation form logic
        // (rater/storage/CSV plumbing comes from form_core.js)
        const DIMENSIONS = CFG.dims;

        function updateDimNotes(dimId, scoreVal) {
            const container = document.getElementById(`dimNotes_${dimId}`);
            if (!container) return;
            if (scoreVal === 'NA' || parseInt(scoreVal) <= 2) {
                container.style.display = 'block';
            } else {
                container.style.display = 'none';
            }
        }

        // Score selection handlers
        document.querySelectorAll('.score-option').forEach(opt => {
            opt.addEventListener('click', () => {
                const dimId = opt.dataset.dimension;
                const scoreVal = opt.dataset.score;
                opt.closest('.score-options').querySelectorAll('.score-option').forEach(o => {
                    o.classList.remove('selected');
                    o.querySelector('input[type="radio"]').checked = false;
                });
                opt.classList.add('selected');
                opt.querySelector('input[type="radio"]').checked = true;
                updateDimNotes(dimId, scoreVal);
                saveToLocalStorage();
                updateProgress();
                showStatus('Auto-saved');
            });
        });

        // Auto-save per-dimension notes on blur
        document.querySelectorAll('.dim-notes-input').forEach(el => {
            el.addEventListener('blur', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
        });

        function getRatedCount() {
            let rated = 0;
            DIMENSIONS.forEach(dim => {
                const checked = document.querySelector(`input[name="dim_${dim.id}"]:checked`);
                if (checked) rated++;
            });
            return rated;
        }

        function updateProgress() {
            const rated = getRatedCount();
            const total = DIMENSIONS.length;
            const el = document.getElementById('ratingProgress');
            if (el) {
                el.textContent = '';
                const count = document.createElement('span');
                count.className = 'count';
                count.textContent = String(rated);
                el.appendChild(count);
                el.appendChild(document.createTextNode(` / ${total} dimensions rated`));
            }
            // Enable/disable mark-complete checkbox
            const completeCheckbox = document.getElementById('markComplete');
            const completeLabel = completeCheckbox?.closest('.complete-checkbox');
            if (rated < total) {
                // Don't let a completed flag survive incomplete ratings.
                if (completeCheckbox.checked) {
                    completeCheckbox.checked = false;
                    saveToLocalStorage();
                }
                completeCheckbox.disabled = true;
                if (completeLabel) {
                    completeLabel.style.opacity = '0.5';
                    completeLabel.style.cursor = 'not-allowed';
                    completeLabel.title = `Rate all ${total} dimensions before marking complete (${total - rated} remaining)`;
                }
            } else {
                completeCheckbox.disabled = false;
                if (completeLabel) {
                    completeLabel.style.opacity = '1';
                    completeLabel.style.cursor = 'pointer';
                    completeLabel.title = '';
                }
            }
        }

        function initializeForm() {
            const stored = getStoredAnnotations();
            const saved = stored[ANNOTATION_KEY];

            if (saved) {
                DIMENSIONS.forEach(dim => {
                    const val = saved[`dim_${dim.id}`];
                    if (val) {
                        const radio = document.querySelector(`input[name="dim_${dim.id}"][value="${val}"]`);
                        if (radio) {
                            radio.checked = true;
                            radio.closest('.score-option').classList.add('selected');
                        }
                        updateDimNotes(dim.id, val);
                    }
                    const notesVal = saved[`dim_${dim.id}_notes`];
                    if (notesVal) {
                        const notesInput = document.getElementById(`dimNotesInput_${dim.id}`);
                        if (notesInput) notesInput.value = notesVal;
                    }
                });
                const comments = document.getElementById('freeTextComments');
                if (comments && saved.free_text_comments) {
                    comments.value = saved.free_text_comments;
                }
                const complete = document.getElementById('markComplete');
                if (complete) {
                    complete.checked = saved.completed || false;
                }
                showStatus(saved.completed ? 'Loaded completed annotation' : 'Loaded saved annotation');
            }
            updateProgress();
        }

        function generateAnnotation() {
            const ann = {
                id: `annotation_${Date.now().toString(36)}`,
                simulation_id: annotationMeta.simulation_id,
                task_id: annotationMeta.task_id,
                trial: annotationMeta.trial,
                rater: RATER_NAME,
                batch: BATCH_NAME,
                task_type: 'user_realism',
                created_at: new Date().toISOString(),
                completed: document.getElementById('markComplete').checked,
            };

            DIMENSIONS.forEach(dim => {
                const checked = document.querySelector(`input[name="dim_${dim.id}"]:checked`);
                ann[`dim_${dim.id}`] = checked ? (checked.value === 'NA' ? 'NA' : parseInt(checked.value)) : null;
                const notesInput = document.getElementById(`dimNotesInput_${dim.id}`);
                ann[`dim_${dim.id}_notes`] = notesInput ? notesInput.value : '';
            });

            ann.free_text_comments = document.getElementById('freeTextComments')?.value || '';

            return ann;
        }

        initializeForm();

        // Auto-save text fields on blur
        const commentsEl = document.getElementById('freeTextComments');
        if (commentsEl) {
            commentsEl.addEventListener('blur', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
        }

        document.getElementById('markComplete').addEventListener('change', () => {
            const isComplete = document.getElementById('markComplete').checked;
            if (isComplete && getRatedCount() < DIMENSIONS.length) {
                document.getElementById('markComplete').checked = false;
                showStatus(`Please rate all ${DIMENSIONS.length} dimensions first`, true);
                return;
            }
            if (isComplete) {
                const missing = [];
                DIMENSIONS.forEach(dim => {
                    const checked = document.querySelector(`input[name="dim_${dim.id}"]:checked`);
                    if (checked && (checked.value === 'NA' || parseInt(checked.value) <= 2)) {
                        const notes = document.getElementById(`dimNotesInput_${dim.id}`);
                        if (!notes || !notes.value.trim()) {
                            missing.push(dim.name);
                        }
                    }
                });
                if (missing.length > 0) {
                    document.getElementById('markComplete').checked = false;
                    showStatus(`Please add notes for low-rated or N/A dimensions: ${missing.join(', ')}`, true);
                    return;
                }
            }
            saveToLocalStorage();
            if (isComplete) {
                autoBackupCSV();
                showStatus('Marked as complete! Backup CSV downloaded.');
            } else {
                showStatus('Marked as in progress');
            }
        });

        wireAnnotationButtons(saveToLocalStorage);
