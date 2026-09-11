const CFG = window.PACKET_CONFIG;
const RATER_KEY = `tau2_ann_${CFG.batch_id}_rater`;

function getRaterName() { return localStorage.getItem(RATER_KEY) || ''; }
function storageKey() { return `tau2_ann_${CFG.batch_id}_calibration_decisions_${getRaterName()}`; }
function getStored() { try { return JSON.parse(localStorage.getItem(storageKey()) || '{}'); } catch (_) { return {}; } }
function setStored(value) { localStorage.setItem(storageKey(), JSON.stringify(value)); }
function promptRaterName() { document.getElementById('raterModal').classList.add('active'); document.getElementById('raterNameInput').focus(); }
function submitRaterName() {
    const name = document.getElementById('raterNameInput').value.trim();
    if (!name) return;
    localStorage.setItem(RATER_KEY, name);
    document.getElementById('raterModal').classList.remove('active');
    document.getElementById('raterDisplay').textContent = name;
    restore();
}

function collect(card) {
    const old = getStored()[card.dataset.candidateId] || {};
    const selected = card.querySelector('[data-decision][aria-pressed="true"]');
    return {
        decision: selected?.dataset.decision || '',
        reclassified: card.querySelector('[data-reclassify]').value,
        note: card.querySelector('[data-note]').value,
        completed: card.querySelector('[data-completed]').checked,
        created_at: old.created_at || new Date().toISOString()
    };
}

function update(card, row) {
    const message = !row.decision ? 'Select a decision.'
        : (row.reclassified && row.decision !== 'rejected')
            ? 'A reclassified finding must be Rejected here.'
            : '';
    const complete = card.querySelector('[data-completed]');
    complete.disabled = Boolean(message);
    if (message && complete.checked) {
        complete.checked = false;
        row.completed = false;
    }
    card.querySelector('[data-blocker]').textContent = message || 'Ready.';
    card.querySelector('[data-status]').textContent = row.completed ? 'Complete' : (row.decision ? 'In progress' : 'Not started');
    card.querySelector('[data-status]').classList.toggle('complete', Boolean(row.completed));
}

function progress() {
    const stored = getStored();
    document.getElementById('completedCount').textContent = Object.values(stored).filter(row => row.completed).length;
    document.querySelectorAll('.ca-decisions').forEach(section => {
        const cards = [...section.querySelectorAll('.ca-candidate')];
        const decided = cards.filter(card => stored[card.dataset.candidateId]?.completed).length;
        const status = document.querySelector(`[data-call-status][data-clip="${section.dataset.clip}"]`);
        if (status) {
            status.textContent = `${decided} / ${cards.length} decided`;
            status.classList.toggle('complete', cards.length > 0 && decided === cards.length);
        }
    });
}

function save(card) {
    const stored = getStored();
    const row = collect(card);
    update(card, row);
    stored[card.dataset.candidateId] = row;
    setStored(stored);
    progress();
}

function restore() {
    const stored = getStored();
    document.querySelectorAll('.ca-candidate').forEach(card => {
        const row = stored[card.dataset.candidateId] || {};
        card.querySelectorAll('[data-decision]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.decision === row.decision)));
        card.querySelector('[data-reclassify]').value = row.reclassified || '';
        card.querySelector('[data-note]').value = row.note || '';
        card.querySelector('[data-completed]').checked = Boolean(row.completed);
        update(card, collect(card));
    });
    progress();
}

function escapeCSV(value) {
    let text = value === null || value === undefined ? '' : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
    if (/[",\n\r]/.test(text)) text = '"' + text.replace(/"/g, '""') + '"';
    return text;
}
function exportRow(values) { return CFG.csv_headers.map(header => escapeCSV(values[header] ?? '')).join(','); }
function exportCSV() {
    const stored = getStored();
    const rows = CFG.candidates.map(candidate => {
        const state = stored[candidate.candidate_id] || {};
        const flagged = candidate.agent_turn;
        return exportRow({
            batch: CFG.batch_name,
            rater: getRaterName(),
            clip_id: candidate.clip_id,
            simulation_id: candidate.sim_id,
            task_id: candidate.task_id,
            language: candidate.language,
            judge: candidate.judge,
            factor_id: candidate.factor_id,
            evaluation_level: candidate.evaluation_level,
            judge_verdict: candidate.judge_verdict,
            candidate_id: candidate.candidate_id,
            adjudication_decision: state.decision || '',
            reclassified_factor: state.reclassified || '',
            agent_turn_index: flagged?.index ?? '',
            agent_turn_id: flagged?.turn_id || '',
            agent_text: flagged?.text || '',
            preceding_customer_text: flagged?.preceding_customer_text || '',
            note: state.note || '',
            provenance_json: JSON.stringify({ packet_batch_id: CFG.batch_id, packet_version: CFG.packet_version, sidecar_batch_id: CFG.sidecar_batch_id, source: 'judge_calibration_adjudication' }),
            completed: Boolean(state.completed),
            created_at: state.created_at || ''
        });
    });
    const content = [CFG.csv_headers.map(escapeCSV).join(','), ...rows].join('\r\n') + '\r\n';
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${CFG.batch_name}_${getRaterName() || 'unnamed'}_decisions.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

function setAudioMode(call, mode) {
    const audio = call.querySelector('audio');
    if (!audio) return;
    if (audio.dataset.currentMode !== mode) {
        audio.pause();
        audio.src = mode === 'agent_only' ? audio.dataset.agentSrc : audio.dataset.fullSrc;
        audio.dataset.currentMode = mode;
        audio.load();
    }
    call.querySelectorAll('.rp-mode-toggle [data-mode]').forEach(button =>
        button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
    const description = call.querySelector('[data-audio-description]');
    if (description) description.textContent = mode === 'agent_only'
        ? `Agent only · ${audio.dataset.agentDuration}`
        : `Full conversation · ${audio.dataset.fullDuration}`;
}

function seekAndPlay(audio, seconds) {
    // Audio elements ship preload="none" (one page carries ~100 calls); a
    // seek before metadata arrives is silently dropped, so defer it.
    const go = () => { audio.currentTime = seconds; audio.play(); };
    if (audio.readyState >= 1) go();
    else {
        audio.addEventListener('loadedmetadata', go, { once: true });
        audio.load();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const name = getRaterName();
    if (!name) promptRaterName();
    else document.getElementById('raterDisplay').textContent = name;

    document.querySelectorAll('.ca-call').forEach(call => {
        if (!call.querySelector('audio')) return;
        call.querySelectorAll('.rp-mode-toggle [data-mode]').forEach(button =>
            button.addEventListener('click', () => setAudioMode(call, button.dataset.mode)));
        call.querySelectorAll('.clickable-time').forEach(cell => cell.addEventListener('click', () => {
            // Transcript times are full-call time; the agent-only
            // concatenation runs on its own clock.
            setAudioMode(call, 'full_conversation');
            seekAndPlay(call.querySelector('audio'), Number(cell.closest('tr').dataset.startTime || 0));
        }));
    });

    document.querySelectorAll('.ca-candidate').forEach(card => {
        card.querySelectorAll('[data-decision]').forEach(button => button.addEventListener('click', () => {
            card.querySelectorAll('[data-decision]').forEach(item => item.setAttribute('aria-pressed', 'false'));
            button.setAttribute('aria-pressed', 'true');
            // Confirmed contradicts a "different category" pick: clear it.
            if (button.dataset.decision === 'confirmed') card.querySelector('[data-reclassify]').value = '';
            save(card);
        }));
        card.querySelector('[data-reclassify]').addEventListener('change', event => {
            // Naming another category IS a rejection of this row's factor.
            if (event.target.value) {
                card.querySelectorAll('[data-decision]').forEach(item =>
                    item.setAttribute('aria-pressed', String(item.dataset.decision === 'rejected')));
            }
            save(card);
        });
        card.querySelectorAll('input').forEach(element => element.addEventListener(element.type === 'text' ? 'input' : 'change', () => save(card)));
    });
    document.getElementById('raterDisplay').addEventListener('click', promptRaterName);
    restore();
});
