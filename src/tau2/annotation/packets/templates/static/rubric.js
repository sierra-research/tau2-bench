const CFG = window.PACKET_CONFIG;
const RATER_KEY = `tau2_ann_${CFG.batch_id}_rater`;

function getRaterName() { return localStorage.getItem(RATER_KEY) || ''; }
function storageKey() { return `tau2_ann_${CFG.batch_id}_rubric_${getRaterName()}`; }
function getStored() {
    try { return JSON.parse(localStorage.getItem(storageKey()) || '{}'); }
    catch (_) { return {}; }
}
function setStored(value) { localStorage.setItem(storageKey(), JSON.stringify(value)); }
function promptRaterName() {
    document.getElementById('raterModal').classList.add('active');
    document.getElementById('raterNameInput').focus();
}
function submitRaterName() {
    const name = document.getElementById('raterNameInput').value.trim();
    if (!name) return;
    localStorage.setItem(RATER_KEY, name);
    document.getElementById('raterModal').classList.remove('active');
    document.getElementById('raterDisplay').textContent = name;
    restoreAll();
}
function questionKey(section, factorId) { return `${section}:${factorId}`; }
function rowKey(row) { return questionKey(row.dataset.sectionId, row.dataset.factorId); }
function isUtteranceRow(row) { return row.dataset.evaluationLevel === 'utterance'; }
function parseSelected(row) {
    const input = row.querySelector('[data-selected-turn-indices]');
    if (!input) return [];
    try { return JSON.parse(input.value || '[]'); }
    catch (_) { return []; }
}
function writeSelected(row, values) {
    const input = row.querySelector('[data-selected-turn-indices]');
    if (!input) return;
    const selected = [...new Set(values.map(Number))].sort((a, b) => a - b);
    input.value = JSON.stringify(selected);
    const summary = row.querySelector('[data-selected-turns]');
    if (summary) summary.textContent = selected.length ? `Selected agent turns: ${selected.join(', ')}` : 'No turns selected.';
}
function selectedSeverity(row) {
    const pressed = row.querySelector('[data-severity][aria-pressed="true"]');
    return pressed ? pressed.dataset.severity : '';
}
function writeSeverity(row, severity) {
    row.querySelectorAll('[data-severity]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.severity === severity)));
}
// Whether this call is still in phase 1 (blind) of a two-phase page. Blind
// builds have no phases at all: every call behaves as already "revealed".
function isBlindPhase(state) { return Boolean(CFG.judge_visible) && state.phase !== 'revealed'; }

function collectCall(card) {
    const prior = getStored()[card.dataset.clipId] || {};
    const answers = {};
    const nativeness = {};
    card.querySelectorAll('.rp-question').forEach(row => {
        if (row.dataset.answerKind === 'nativeness') {
            const selected = row.querySelector('[data-native-label][aria-pressed="true"]');
            nativeness[row.dataset.factorId] = {
                label: selected ? selected.dataset.nativeLabel : '',
                severity: row.querySelector('[data-native-severity]')?.value || '',
                note: row.querySelector('[data-native-note]').value,
                selected_turn_indices: parseSelected(row)
            };
            return;
        }
        const selected = row.querySelector('[data-answer][aria-pressed="true"]');
        answers[rowKey(row)] = {
            answer: selected ? selected.dataset.answer : '',
            evidence: row.querySelector('[data-evidence]').value,
            severity: selectedSeverity(row),
            selected_turn_indices: parseSelected(row)
        };
    });
    const selectedMode = card.querySelector('[data-mode][aria-pressed="true"]');
    const state = {
        evidence_mode: selectedMode ? selectedMode.dataset.mode : 'full_conversation',
        audio_consulted: Boolean(prior.audio_consulted),
        active_factor_key: card.dataset.activeFactorKey || '',
        answers,
        nativeness,
        notes: card.querySelector('[data-notes]').value,
        completed: card.querySelector('[data-field="completed"]').checked,
        created_at: prior.created_at || new Date().toISOString()
    };
    if (CFG.judge_visible) {
        // Two-phase state. pre_reveal is IMMUTABLE once phase 2 starts: it is
        // written exactly once by lockBlindAnswers and only copied here.
        state.phase = prior.phase === 'revealed' ? 'revealed' : 'blind';
        if (prior.pre_reveal) state.pre_reveal = prior.pre_reveal;
    }
    return state;
}

// Display number + id ("3.6 — gender_agreement") so a blocker message names
// the exact on-page question the rater must go back to.
function questionLabel(question) {
    return question.number ? `${question.number} — ${question.id}` : question.id;
}
function nativenessBlocker(question, verdict) {
    if (!verdict?.label) return `Label ${questionLabel(question)}.`;
    if (verdict.label !== 'violation') return '';
    if (question.evaluation_level === 'call' && !verdict.severity) return `Set severity for ${questionLabel(question)}.`;
    if (question.evaluation_level === 'utterance' && !verdict.selected_turn_indices?.length) {
        return `Select at least one agent turn for ${questionLabel(question)}.`;
    }
    return '';
}
function positiveAnswer(question, answer) {
    if (question.answer_kind === 'binary') return answer === 'yes';
    if (question.answer_kind === 'likert4') return ['1', '2'].includes(answer);
    return ['1', '2', '3'].includes(answer);
}
function answerBlocker(question, verdict) {
    if (!verdict?.answer) return `Answer ${questionLabel(question)}.`;
    if (positiveAnswer(question, verdict.answer) && !(verdict.evidence || '').trim()) {
        return `Evidence required for ${questionLabel(question)}: say what you saw or heard.`;
    }
    if (question.answer_kind !== 'binary' || verdict.answer !== 'yes') return '';
    if (question.severity_scale === 'minor_major' && !verdict.severity) {
        return `Set minor/major severity for ${questionLabel(question)}.`;
    }
    if (question.evaluation_level === 'utterance' && !verdict.selected_turn_indices?.length) {
        return `Select at least one agent turn for ${questionLabel(question)}.`;
    }
    return '';
}
function completionBlocker(row) {
    for (const question of CFG.questions) {
        const blocker = question.answer_kind === 'nativeness'
            ? nativenessBlocker(question, row.nativeness[question.id])
            : answerBlocker(question, row.answers[questionKey(question.section, question.id)]);
        if (blocker) return blocker;
    }
    if (CFG.questions.some(question => question.section === 'audio') && !row.audio_consulted) {
        return 'Play the agent-only audio for the audio-quality section.';
    }
    return '';
}
function answerNeedsEvidence(row, answer) {
    if (row.dataset.answerKind === 'binary') return answer === 'yes';
    if (row.dataset.answerKind === 'likert4') return ['1', '2'].includes(answer);
    return ['1', '2', '3'].includes(answer);
}
function answerCount(row) {
    return Object.values(row.answers).filter(value => value.answer).length
        + Object.values(row.nativeness).filter(value => value.label).length;
}
function updateCardState(card, row) {
    const blocker = completionBlocker(row);
    const blind = isBlindPhase(row);
    card.classList.toggle('phase-revealed', Boolean(CFG.judge_visible) && !blind);
    const lockButton = card.querySelector('[data-lock-blind]');
    if (lockButton) {
        lockButton.hidden = !blind;
        lockButton.disabled = Boolean(blocker) || !blind;
    }
    const phaseBadge = card.querySelector('[data-phase-badge]');
    if (phaseBadge) phaseBadge.textContent = blind ? 'Phase 1 of 2 — blind' : 'Phase 2 of 2 — findings revealed';
    const confirmLabel = card.querySelector('[data-confirm-label]');
    if (confirmLabel) confirmLabel.hidden = blind;
    const completed = card.querySelector('[data-field="completed"]');
    completed.disabled = Boolean(blocker) || blind;
    if ((blocker || blind) && completed.checked) {
        completed.checked = false;
        row.completed = false;
    }
    card.querySelector('[data-blocker]').textContent = blocker
        || (blind ? 'Ready to lock blind answers and reveal judge findings.' : 'Ready to mark complete.');
    const answered = answerCount(row);
    card.querySelector('[data-answered]').textContent = String(answered);
    card.querySelectorAll('[data-section]').forEach(section => {
        const count = Array.from(section.querySelectorAll('.rp-question')).filter(question => {
            if (question.dataset.answerKind === 'nativeness') {
                return Boolean(row.nativeness[question.dataset.factorId]?.label);
            }
            return Boolean(row.answers[rowKey(question)]?.answer);
        }).length;
        section.querySelector('[data-section-answered]').textContent = String(count);
    });
    const status = card.querySelector('[data-status]');
    status.textContent = row.completed ? 'Complete'
        : (CFG.judge_visible && !blind ? 'Revealed — reviewing'
            : (answered ? 'In progress' : 'Not started'));
    status.classList.toggle('complete', Boolean(row.completed));
    document.querySelector(`[data-go="${card.dataset.clipId}"]`).classList.toggle('complete', Boolean(row.completed));
}
function saveCard(card) {
    const stored = getStored();
    const row = collectCall(card);
    updateCardState(card, row);
    stored[card.dataset.clipId] = row;
    setStored(stored);
    updateProgress();
}
function lockBlindAnswers(card) {
    // Phase 1 → 2: freeze the blind answers as the pre-reveal record, then
    // reveal the judge findings. There is deliberately no way back.
    const stored = getStored();
    const row = collectCall(card);
    if (row.phase === 'revealed' || completionBlocker(row)) return;
    // Snapshot EVERY answer-bearing or call-level mutable field that rides on
    // exported rows: a phase-2 edit must never reach a pre_reveal row.
    row.pre_reveal = {
        answers: JSON.parse(JSON.stringify(row.answers)),
        nativeness: JSON.parse(JSON.stringify(row.nativeness)),
        notes: row.notes,
        evidence_mode: row.evidence_mode,
        audio_consulted: row.audio_consulted,
        locked_at: new Date().toISOString()
    };
    row.phase = 'revealed';
    updateCardState(card, row);
    stored[card.dataset.clipId] = row;
    setStored(stored);
    updateProgress();
}

function setEvidenceMode(card, mode, persist = true) {
    card.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
    card.querySelectorAll('[data-evidence-view]').forEach(view => view.classList.toggle('active', view.dataset.evidenceView === mode));
    const full = mode === 'full_conversation';
    card.querySelector('[data-evidence-title]').textContent = full ? 'Full conversation trace' : 'Agent-only transcript';
    card.querySelector('[data-mode-badge]').textContent = full ? 'Full conversation' : 'Agent only';
    const audio = card.querySelector('audio');
    const desired = full ? audio.dataset.fullSrc : audio.dataset.agentSrc;
    const duration = full ? audio.dataset.fullDuration : audio.dataset.agentDuration;
    card.querySelector('[data-audio-description]').textContent = `${full ? 'Full conversation' : 'Agent only'} · ${duration}`;
    if (audio.dataset.currentMode !== mode) {
        audio.pause();
        audio.src = desired;
        audio.dataset.currentMode = mode;
        audio.load();
    }
    refreshFloatAudio();
    if (persist) saveCard(card);
}
// ONE turn-selection mechanism for every utterance-level question, whatever
// its answer kind: the active question is the turn-click target.
function setActiveFactor(card, question, persist = true) {
    const active = question && isUtteranceRow(question) ? question : null;
    card.dataset.activeFactorKey = active ? rowKey(active) : '';
    card.querySelectorAll('.rp-question[data-evaluation-level="utterance"]').forEach(row => row.classList.toggle('active-factor', row === active));
    const selected = active ? parseSelected(active) : [];
    card.querySelectorAll('[data-agent-turn]').forEach(turn => {
        turn.classList.toggle('selectable', Boolean(active));
        turn.classList.toggle('selected', selected.includes(Number(turn.dataset.turnIndex)));
    });
    if (active) setEvidenceMode(card, 'agent_only', false);
    if (persist) saveCard(card);
}
function questionByKey(card, key) {
    if (!key) return null;
    const separator = key.indexOf(':');
    const section = key.slice(0, separator);
    const factorId = key.slice(separator + 1);
    return card.querySelector(`.rp-question[data-section-id="${section}"][data-factor-id="${factorId}"]`);
}

function restoreAll() {
    const stored = getStored();
    document.querySelectorAll('.rp-call').forEach(card => {
        const state = stored[card.dataset.clipId] || { answers: {}, nativeness: {} };
        setEvidenceMode(card, state.evidence_mode || 'full_conversation', false);
        card.querySelector('[data-notes]').value = state.notes || '';
        card.querySelector('[data-field="completed"]').checked = Boolean(state.completed);
        card.querySelectorAll('.rp-question').forEach(row => {
            if (row.dataset.answerKind === 'nativeness') {
                const value = state.nativeness?.[row.dataset.factorId] || {};
                row.querySelectorAll('[data-native-label]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.nativeLabel === value.label)));
                const severity = row.querySelector('[data-native-severity]');
                if (severity) severity.value = value.severity || '';
                row.querySelector('[data-native-note]').value = value.note || '';
                writeSelected(row, value.selected_turn_indices || []);
                row.classList.toggle('show-native-detail', value.label === 'violation');
                return;
            }
            const value = state.answers?.[rowKey(row)] || {};
            row.querySelectorAll('[data-answer]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.answer === value.answer)));
            row.querySelector('[data-evidence]').value = value.evidence || '';
            writeSeverity(row, value.answer === 'yes' ? value.severity || '' : '');
            writeSelected(row, value.answer === 'yes' ? value.selected_turn_indices || [] : []);
            row.classList.toggle('show-evidence', answerNeedsEvidence(row, value.answer));
            row.classList.toggle('show-violation-detail', row.dataset.answerKind === 'binary' && value.answer === 'yes');
        });
        setActiveFactor(card, questionByKey(card, state.active_factor_key || ''), false);
        updateCardState(card, collectCall(card));
    });
    updateProgress();
}
function updateProgress() {
    document.getElementById('completedCount').textContent = Object.values(getStored()).filter(row => row.completed).length;
}
function showCall(clipId) {
    document.querySelectorAll('audio').forEach(audio => audio.pause());
    document.querySelectorAll('.rp-call').forEach(card => card.classList.toggle('active', card.id === clipId));
    document.querySelectorAll('.rp-call-pill').forEach(pill => pill.classList.toggle('active', pill.dataset.go === clipId));
    window.scrollTo({ top: document.querySelector('.rp-call-nav').offsetTop - 12, behavior: 'smooth' });
    refreshFloatAudio();
}

// Floating controls drive the ACTIVE call's audio element, so raters can
// pause/play from anywhere on the page instead of scrolling back to the
// audio panel.
function activeCallAudio() {
    const card = document.querySelector('.rp-call.active');
    return card ? card.querySelector('audio') : null;
}
function formatClock(seconds) {
    if (!Number.isFinite(seconds)) return '0:00';
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}
function refreshFloatAudio() {
    const bar = document.querySelector('[data-float-audio]');
    const audio = activeCallAudio();
    if (!bar || !audio) return;
    bar.querySelector('[data-float-mode]').textContent =
        audio.dataset.currentMode === 'agent_only' ? 'Agent only' : 'Full conversation';
    bar.querySelector('[data-float-time]').textContent = formatClock(audio.currentTime);
    const toggle = bar.querySelector('[data-float-toggle]');
    toggle.textContent = audio.paused ? 'Play' : 'Pause';
    toggle.setAttribute('aria-pressed', String(!audio.paused));
}
function wireFloatAudio() {
    const bar = document.querySelector('[data-float-audio]');
    if (!bar) return;
    bar.querySelector('[data-float-toggle]').addEventListener('click', () => {
        const audio = activeCallAudio();
        if (!audio) return;
        if (audio.paused) audio.play(); else audio.pause();
    });
    bar.querySelector('[data-float-back]').addEventListener('click', () => {
        const audio = activeCallAudio();
        if (!audio) return;
        audio.currentTime = Math.max(0, audio.currentTime - 10);
        refreshFloatAudio();
    });
    document.querySelectorAll('audio').forEach(audio => {
        ['play', 'pause', 'timeupdate', 'seeked', 'loadedmetadata', 'emptied'].forEach(event =>
            audio.addEventListener(event, refreshFloatAudio));
    });
    refreshFloatAudio();
}

function escapeCSV(value) {
    let text = value === null || value === undefined ? '' : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
    if (/[",\n\r]/.test(text)) text = '"' + text.replace(/"/g, '""') + '"';
    return text;
}
function exportRow(headers, values) { return headers.map(header => escapeCSV(values[header] ?? '')).join(','); }
function download(content, mime, filename) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}
// The answer layers one call exports. Blind packets export ONE phase-less
// layer. Two-phase packets export BOTH layers for every question: the locked
// pre-reveal record and the (possibly revised) post-reveal record — identical
// when the rater changed nothing after the reveal. A call not yet locked
// exports its live answers under both phases, as an incomplete draft.
// `completed` and `created_at` stay call-level on BOTH layers by design:
// completed is the phase-2 confirm that makes the whole call's rows count,
// and created_at is the first-save timestamp (never rewritten after).
function layerFields(source, state) {
    // Call-level fields exported on every row of a layer. The pre_reveal
    // layer reads them from the lock-time snapshot so phase-2 edits to the
    // notes / evidence mode / audio flag can never contaminate blind rows.
    return {
        answers: source.answers || {},
        nativeness: source.nativeness || {},
        notes: source.notes ?? state.notes,
        evidence_mode: source.evidence_mode ?? state.evidence_mode,
        audio_consulted: source.audio_consulted ?? state.audio_consulted
    };
}
function exportLayers(state) {
    if (!CFG.judge_visible) {
        return [{ phase: '', ...layerFields(state, state) }];
    }
    return [
        { phase: 'pre_reveal', ...layerFields(state.pre_reveal || state, state) },
        { phase: 'post_reveal', ...layerFields(state, state) }
    ];
}
function rubricRowsForExport() {
    const stored = getStored();
    const rows = [];
    CFG.clip_ids.forEach(clipId => {
        const state = stored[clipId] || { answers: {} };
        const context = CFG.contexts[clipId];
        exportLayers(state).forEach(layer => {
            CFG.questions.filter(question => question.answer_kind !== 'nativeness').forEach(question => {
                const verdict = layer.answers?.[questionKey(question.section, question.id)] || {};
                const violation = question.answer_kind === 'binary' && verdict.answer === 'yes';
                rows.push(exportRow(CFG.rubric_csv_headers, {
                    batch: CFG.batch_name, rater: getRaterName(), clip_id: clipId,
                    language: CFG.language, agent_gender: context.agent_gender,
                    caller_gender: context.caller_gender, section: question.section,
                    factor_id: question.id, question: question.question,
                    answer: verdict.answer || '',
                    severity: violation ? verdict.severity || '' : '',
                    evidence: verdict.evidence || '',
                    selected_turn_indices: violation && question.evaluation_level === 'utterance'
                        ? (verdict.selected_turn_indices || []).join(', ') : '',
                    is_custom: false, evidence_mode: layer.evidence_mode || 'full_conversation',
                    audio_consulted: Boolean(layer.audio_consulted), notes: layer.notes || '',
                    phase: layer.phase,
                    completed: Boolean(state.completed), created_at: state.created_at || ''
                }));
            });
        });
    });
    return rows;
}
function nativenessRowsForExport() {
    const stored = getStored();
    const rows = [];
    CFG.clip_ids.forEach(clipId => {
        const state = stored[clipId] || { nativeness: {} };
        const call = CFG.calls[clipId];
        exportLayers(state).forEach(layer => {
            CFG.questions.filter(question => question.answer_kind === 'nativeness').forEach(question => {
                const verdict = layer.nativeness?.[question.id] || {};
                let selectedTurns = [null];
                if (verdict.label === 'violation' && question.evaluation_level === 'utterance') {
                    selectedTurns = (verdict.selected_turn_indices || []).map(index => call.agent_turns.find(turn => turn.index === index) || null);
                    if (!selectedTurns.length) selectedTurns = [null];
                }
                selectedTurns.forEach(turn => rows.push(exportRow(CFG.nativeness_csv_headers, {
                    batch: CFG.batch_name, rater: getRaterName(), clip_id: clipId,
                    simulation_id: call.simulation_id, task_id: call.task_id,
                    language: CFG.language, factor_id: question.id,
                    evaluation_level: question.evaluation_level,
                    annotation_label: verdict.label || '', candidate_id: '',
                    adjudication_decision: '', agent_turn_index: turn?.index ?? '',
                    agent_turn_id: turn?.turn_id || '', agent_text: turn?.text || '',
                    preceding_customer_text: turn?.preceding_customer_text || '',
                    severity: verdict.label === 'violation' && question.evaluation_level === 'call' ? verdict.severity || '' : '',
                    note: verdict.note || '', provenance_json: JSON.stringify({
                        packet_batch_id: CFG.batch_id,
                        packet_version: CFG.packet_version,
                        source: CFG.judge_visible ? 'judge_visible_review' : 'blind_combined_packet',
                        ...(CFG.judge_visible ? { judge_visible: true, phase: layer.phase } : {})
                    }), phase: layer.phase,
                    completed: Boolean(state.completed), created_at: state.created_at || ''
                })));
            });
        });
    });
    return rows;
}
function confirmDraftExport() {
    const missing = CFG.clip_ids.length - Object.values(getStored()).filter(row => row.completed).length;
    return !missing || window.confirm(`${missing} calls are not complete. Export the draft anyway?`);
}
function exportFilename(suffix) {
    return `${CFG.batch_name}_${getRaterName() || 'unnamed'}_${suffix}`;
}
function rubricCSVContent() {
    return [CFG.rubric_csv_headers.map(escapeCSV).join(','), ...rubricRowsForExport()].join('\r\n') + '\r\n';
}
function nativenessCSVContent() {
    return [CFG.nativeness_csv_headers.map(escapeCSV).join(','), ...nativenessRowsForExport()].join('\r\n') + '\r\n';
}
function hasNativenessQuestions() {
    return CFG.questions.some(question => question.answer_kind === 'nativeness');
}
function exportRubricCSV() {
    if (!confirmDraftExport()) return;
    download(rubricCSVContent(), 'text/csv;charset=utf-8', exportFilename('overall_interaction_audio.csv'));
}
function exportNativenessCSV() {
    if (!confirmDraftExport()) return;
    download(nativenessCSVContent(), 'text/csv;charset=utf-8', exportFilename('nativeness.csv'));
}
// One click, every deliverable: both CSVs (the nativeness file only when the
// instrument has nativeness questions — the native control has none) plus
// the JSON backup. Downloads are staggered because back-to-back programmatic
// anchor clicks can be dropped, and the browser may ask once to allow
// multiple downloads.
function exportAll() {
    if (!confirmDraftExport()) return;
    const files = [
        () => download(rubricCSVContent(), 'text/csv;charset=utf-8', exportFilename('overall_interaction_audio.csv')),
        ...(hasNativenessQuestions()
            ? [() => download(nativenessCSVContent(), 'text/csv;charset=utf-8', exportFilename('nativeness.csv'))]
            : []),
        () => download(JSON.stringify(getStored(), null, 2), 'application/json', exportFilename('backup.json'))
    ];
    files.forEach((fire, index) => setTimeout(fire, index * 400));
}
function downloadBackup() {
    download(JSON.stringify(getStored(), null, 2), 'application/json', exportFilename('backup.json'));
}

document.addEventListener('DOMContentLoaded', () => {
    const name = getRaterName();
    if (!name) promptRaterName();
    else document.getElementById('raterDisplay').textContent = name;
    document.querySelectorAll('.rp-call').forEach((card, index, cards) => {
        card.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => setEvidenceMode(card, button.dataset.mode)));
        card.querySelectorAll('[data-answer]').forEach(button => button.addEventListener('click', () => {
            const question = button.closest('.rp-question');
            question.querySelectorAll('[data-answer]').forEach(item => item.setAttribute('aria-pressed', 'false'));
            button.setAttribute('aria-pressed', 'true');
            question.classList.toggle('show-evidence', answerNeedsEvidence(question, button.dataset.answer));
            const violation = question.dataset.answerKind === 'binary' && button.dataset.answer === 'yes';
            question.classList.toggle('show-violation-detail', violation);
            if (!violation) {
                writeSeverity(question, '');
                writeSelected(question, []);
            }
            setActiveFactor(card, violation ? question : null, false);
            saveCard(card);
        }));
        card.querySelectorAll('[data-severity]').forEach(button => button.addEventListener('click', () => {
            writeSeverity(button.closest('.rp-question'), button.dataset.severity);
            saveCard(card);
        }));
        card.querySelectorAll('[data-native-label]').forEach(button => button.addEventListener('click', () => {
            const question = button.closest('.rp-question');
            question.querySelectorAll('[data-native-label]').forEach(item => item.setAttribute('aria-pressed', 'false'));
            button.setAttribute('aria-pressed', 'true');
            const violation = button.dataset.nativeLabel === 'violation';
            question.classList.toggle('show-native-detail', violation);
            if (!violation) {
                const severity = question.querySelector('[data-native-severity]');
                if (severity) severity.value = '';
                writeSelected(question, []);
            }
            if (violation && question.dataset.evaluationLevel === 'call') {
                setEvidenceMode(card, 'full_conversation', false);
            }
            setActiveFactor(card, violation ? question : null, false);
            saveCard(card);
        }));
        // Clicking anywhere non-interactive on an utterance-level question
        // (any answer kind) makes it the turn-click target.
        card.querySelectorAll('.rp-question[data-evaluation-level="utterance"]').forEach(question => {
            question.addEventListener('click', event => {
                if (event.target.closest('button, select, input, textarea, label')) return;
                setActiveFactor(card, question, false);
                saveCard(card);
            });
        });
        card.querySelectorAll('[data-agent-turn]').forEach(turn => turn.addEventListener('click', () => {
            const question = questionByKey(card, card.dataset.activeFactorKey || '');
            if (!question) return;
            const indexValue = Number(turn.dataset.turnIndex);
            const selected = parseSelected(question);
            writeSelected(question, selected.includes(indexValue) ? selected.filter(value => value !== indexValue) : [...selected, indexValue]);
            setActiveFactor(card, question, false);
            saveCard(card);
        }));
        card.querySelectorAll('input, select, textarea').forEach(element => {
            if (element.matches('[data-selected-turn-indices]')) return;
            const eventName = element.matches('input[type="text"], textarea') ? 'input' : 'change';
            element.addEventListener(eventName, () => saveCard(card));
        });
        const lockButton = card.querySelector('[data-lock-blind]');
        if (lockButton) lockButton.addEventListener('click', () => lockBlindAnswers(card));
        card.querySelector('audio').addEventListener('play', () => {
            if (card.querySelector('[data-mode="agent_only"]').getAttribute('aria-pressed') !== 'true') return;
            const stored = getStored();
            const row = collectCall(card);
            row.audio_consulted = true;
            updateCardState(card, row);
            stored[card.dataset.clipId] = row;
            setStored(stored);
            updateProgress();
        });
        card.querySelectorAll('.clickable-time').forEach(cell => cell.addEventListener('click', () => {
            setEvidenceMode(card, 'full_conversation');
            const audio = card.querySelector('audio');
            audio.currentTime = Number(cell.closest('tr').dataset.startTime || 0);
            audio.play();
        }));
        const interactionSection = card.querySelector('[data-section="interaction"]');
        interactionSection.addEventListener('toggle', () => {
            if (interactionSection.open) setEvidenceMode(card, 'full_conversation');
        });
        const audioSection = card.querySelector('[data-section="audio"]');
        audioSection.addEventListener('toggle', () => {
            if (audioSection.open) setEvidenceMode(card, 'agent_only');
        });
        card.querySelector('[data-previous]').addEventListener('click', () => { if (index > 0) showCall(cards[index - 1].id); });
        card.querySelector('[data-next]').addEventListener('click', () => { if (index + 1 < cards.length) showCall(cards[index + 1].id); });
    });
    document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => showCall(button.dataset.go)));
    document.getElementById('raterDisplay').addEventListener('click', promptRaterName);
    restoreAll();
    wireFloatAudio();
});
