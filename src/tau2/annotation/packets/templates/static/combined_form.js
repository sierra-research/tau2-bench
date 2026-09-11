        // Combined annotation form logic: error findings + audio realism, one pass.
        // (rater/storage/CSV plumbing comes from form_core.js)
        const DIMENSIONS = CFG.dims;

        // Error-findings fields (the "Findings" section).
        const ERROR_FIELDS = ['summaryErrorSource', 'summaryErrorType', 'summaryNotes'];

        // Caller-experience section config (PACKET_CONFIG.experience).
        const EXPERIENCE = CFG.experience || null;

        function getExperienceScore() {
            const checked = document.querySelector('input[name="caller_experience"]:checked');
            return checked ? parseInt(checked.value) : null;
        }

        // The breaking point / factors / notes detail is only meaningful for
        // scores 1-2 ("why was it bad").
        function experienceDetailActive() {
            const score = getExperienceScore();
            return score !== null && score <= 2;
        }

        function updateExperienceDetail() {
            const detail = document.getElementById('experienceDetail');
            if (!detail) return;
            detail.style.display = experienceDetailActive() ? 'block' : 'none';
            const bp = document.querySelector('input[name="experience_breaking_point"]:checked');
            const tickRow = document.getElementById('experienceTickRow');
            if (tickRow) tickRow.style.display = (bp && bp.value === 'tick') ? 'flex' : 'none';
        }

        // Show/hide per-dimension notes based on score
        function updateDimNotes(dimId, scoreVal) {
            const container = document.getElementById(`dimNotes_${dimId}`);
            if (!container) return;
            if (scoreVal === 'NA' || parseInt(scoreVal) <= 2) {
                container.style.display = 'block';
            } else {
                container.style.display = 'none';
            }
        }

        // Score selection handlers (audio section). Scoped to dimension cards:
        // the caller-experience Likert reuses .score-option styling but has
        // its own handler below.
        document.querySelectorAll('.dimension-card[data-dim-id] .score-option').forEach(opt => {
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

        // Auto-save error-findings fields on change/blur
        ERROR_FIELDS.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.addEventListener('change', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
            if (el.tagName === 'TEXTAREA') {
                el.addEventListener('blur', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
            }
        });

        // Caller-experience wiring. The whole label is the click target (the
        // radio-check + change event alone can be swallowed by label
        // forwarding), mirroring the dimension-card handler.
        document.querySelectorAll('#experienceCard .score-option').forEach(opt => {
            opt.addEventListener('click', () => {
                opt.closest('.score-options').querySelectorAll('.score-option').forEach(o => {
                    o.classList.remove('selected');
                    o.querySelector('input[type="radio"]').checked = false;
                });
                opt.classList.add('selected');
                opt.querySelector('input[type="radio"]').checked = true;
                updateExperienceDetail();
                saveToLocalStorage();
                updateProgress();
                showStatus('Auto-saved');
            });
        });
        document.querySelectorAll('input[name="experience_breaking_point"]').forEach(radio => {
            radio.addEventListener('change', () => {
                updateExperienceDetail();
                saveToLocalStorage();
                showStatus('Auto-saved');
            });
        });
        document.querySelectorAll('.exp-factor-check').forEach(box => {
            box.addEventListener('change', () => {
                // A factor's "primary" radio is selectable only while the
                // factor itself is checked; unchecking clears a stale primary.
                const primary = document.querySelector(`input[name="primary_factor"][value="${box.dataset.factor}"]`);
                if (primary) {
                    primary.disabled = !box.checked;
                    if (!box.checked && primary.checked) primary.checked = false;
                }
                saveToLocalStorage();
                showStatus('Auto-saved');
            });
        });
        document.querySelectorAll('input[name="primary_factor"]').forEach(radio => {
            radio.addEventListener('change', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
        });
        const expTickEl = document.getElementById('experienceBreakingTick');
        if (expTickEl) {
            expTickEl.addEventListener('change', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
        }
        const expNotesEl = document.getElementById('experienceNotes');
        if (expNotesEl) {
            expNotesEl.addEventListener('blur', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
        }

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
            // Mark-complete is gated on rating all audio dimensions plus the
            // caller-experience score; findings stay optional.
            const expMissing = EXPERIENCE && getExperienceScore() === null;
            const completeCheckbox = document.getElementById('markComplete');
            const completeLabel = completeCheckbox?.closest('.complete-checkbox');
            if (rated < total || expMissing) {
                // Never let a completed flag survive once ratings are incomplete:
                // uncheck and persist so the index can't show "done" for a partial review.
                if (completeCheckbox.checked) {
                    completeCheckbox.checked = false;
                    saveToLocalStorage();
                }
                completeCheckbox.disabled = true;
                if (completeLabel) {
                    completeLabel.style.opacity = '0.5';
                    completeLabel.style.cursor = 'not-allowed';
                    completeLabel.title = rated < total
                        ? `Rate all ${total} audio dimensions before marking complete (${total - rated} remaining)`
                        : 'Rate the caller experience (section ③) before marking complete';
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
                // Findings
                const src = document.getElementById('summaryErrorSource');
                if (src) src.value = saved.summary_error_source || '';
                const type = document.getElementById('summaryErrorType');
                if (type) type.value = saved.summary_error_type || '';
                const notes = document.getElementById('summaryNotes');
                if (notes) notes.value = saved.summary_notes || '';

                // Audio dimensions
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

                // Caller experience
                if (EXPERIENCE) {
                    if (saved.caller_experience) {
                        const radio = document.querySelector(`input[name="caller_experience"][value="${saved.caller_experience}"]`);
                        if (radio) {
                            radio.checked = true;
                            radio.closest('.score-option').classList.add('selected');
                        }
                    }
                    if (saved.experience_breaking_point) {
                        const bp = document.querySelector(`input[name="experience_breaking_point"][value="${saved.experience_breaking_point}"]`);
                        if (bp) bp.checked = true;
                    }
                    const tickEl = document.getElementById('experienceBreakingTick');
                    if (tickEl && saved.experience_breaking_tick != null) {
                        tickEl.value = saved.experience_breaking_tick;
                    }
                    (saved.experience_factors || []).forEach(fid => {
                        const box = document.querySelector(`.exp-factor-check[data-factor="${fid}"]`);
                        if (box) box.checked = true;
                        const primary = document.querySelector(`input[name="primary_factor"][value="${fid}"]`);
                        if (primary) primary.disabled = false;
                    });
                    if (saved.primary_factor) {
                        const primary = document.querySelector(`input[name="primary_factor"][value="${saved.primary_factor}"]`);
                        if (primary && !primary.disabled) primary.checked = true;
                    }
                    const notesEl = document.getElementById('experienceNotes');
                    if (notesEl && saved.experience_notes) {
                        notesEl.value = saved.experience_notes;
                    }
                    updateExperienceDetail();
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
                task_type: 'voice_review',
                created_at: new Date().toISOString(),
                completed: document.getElementById('markComplete').checked,
                summary_error_source: document.getElementById('summaryErrorSource')?.value || null,
                summary_error_type: document.getElementById('summaryErrorType')?.value || null,
                summary_notes: document.getElementById('summaryNotes')?.value || '',
            };

            DIMENSIONS.forEach(dim => {
                const checked = document.querySelector(`input[name="dim_${dim.id}"]:checked`);
                ann[`dim_${dim.id}`] = checked ? (checked.value === 'NA' ? 'NA' : parseInt(checked.value)) : null;
                const notesInput = document.getElementById(`dimNotesInput_${dim.id}`);
                ann[`dim_${dim.id}_notes`] = notesInput ? notesInput.value : '';
            });

            ann.free_text_comments = document.getElementById('freeTextComments')?.value || '';

            if (EXPERIENCE) {
                ann.caller_experience = getExperienceScore();
                if (experienceDetailActive()) {
                    const bp = document.querySelector('input[name="experience_breaking_point"]:checked');
                    ann.experience_breaking_point = bp ? bp.value : null;
                    const tickVal = document.getElementById('experienceBreakingTick')?.value;
                    ann.experience_breaking_tick =
                        (ann.experience_breaking_point === 'tick' && tickVal !== '' && tickVal != null)
                            ? parseInt(tickVal) : null;
                    ann.experience_factors = Array.from(
                        document.querySelectorAll('.exp-factor-check:checked')
                    ).map(box => box.dataset.factor);
                    const primary = document.querySelector('input[name="primary_factor"]:checked');
                    ann.primary_factor = primary ? primary.value : null;
                    ann.experience_notes = document.getElementById('experienceNotes')?.value || '';
                } else {
                    // Scores 3-4 export a clean row: any detail left over from
                    // an earlier low score stays in the DOM but never ships.
                    ann.experience_breaking_point = null;
                    ann.experience_breaking_tick = null;
                    ann.experience_factors = [];
                    ann.primary_factor = null;
                    ann.experience_notes = '';
                }
            }

            return ann;
        }

        initializeForm();

        // Auto-save free-text comments on blur
        const commentsEl = document.getElementById('freeTextComments');
        if (commentsEl) {
            commentsEl.addEventListener('blur', () => { saveToLocalStorage(); showStatus('Auto-saved'); });
        }

        // Completion gate for the caller-experience section: null = valid,
        // otherwise the message to show. Scores 1-2 must locate the damage,
        // name the contributing factors (one primary), and explain why.
        function validateExperience() {
            const score = getExperienceScore();
            if (score === null) return 'Please rate the caller experience (section ③)';
            if (score > 2) return null;
            const bp = document.querySelector('input[name="experience_breaking_point"]:checked');
            if (!bp) return 'Section ③: pick where the call broke — overall experience or a specific tick';
            if (bp.value === 'tick') {
                const tick = document.getElementById('experienceBreakingTick')?.value;
                if (tick === '' || tick == null) return 'Section ③: give the tick number of the breaking moment';
            }
            const factors = document.querySelectorAll('.exp-factor-check:checked');
            if (factors.length === 0) return 'Section ③: check at least one contributing factor';
            if (!document.querySelector('input[name="primary_factor"]:checked')) {
                return 'Section ③: mark exactly one checked factor as primary';
            }
            const notes = document.getElementById('experienceNotes')?.value;
            if (!notes || !notes.trim()) return 'Section ③: explain why the experience was bad';
            return null;
        }

        document.getElementById('markComplete').addEventListener('change', () => {
            const isComplete = document.getElementById('markComplete').checked;
            if (isComplete && getRatedCount() < DIMENSIONS.length) {
                document.getElementById('markComplete').checked = false;
                showStatus(`Please rate all ${DIMENSIONS.length} audio dimensions first`, true);
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
                if (EXPERIENCE) {
                    const expError = validateExperience();
                    if (expError) {
                        document.getElementById('markComplete').checked = false;
                        showStatus(expError, true);
                        return;
                    }
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
