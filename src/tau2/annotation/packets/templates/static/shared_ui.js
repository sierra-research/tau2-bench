        // Shared page UI: column toggles, LLM-judge error markers, audio controls.
        // (Ported verbatim from the old annotation.js / _generate_shared_ui_js;
        // the only functional edit is reading judge errors from PACKET_CONFIG.)

        // Column toggle functionality
        document.querySelectorAll('.column-toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                const col = btn.dataset.col;
                btn.classList.toggle('hidden');

                // Toggle only table cells (th and td), not the buttons themselves
                document.querySelectorAll(`th[data-col="${col}"], td[data-col="${col}"]`).forEach(cell => {
                    cell.classList.toggle('col-hidden');
                });
            });
        });

        // Auto-collapse empty columns on load
        function checkEmptyColumns() {
            const cols = ['agent-tools', 'user-tools'];

            cols.forEach(col => {
                const cells = document.querySelectorAll(`td[data-col="${col}"]`);
                let isEmpty = true;

                cells.forEach(cell => {
                    // Check if cell has any real content (not just "-")
                    if (cell.textContent.trim() !== '-' && cell.textContent.trim() !== '') {
                        isEmpty = false;
                    }
                });

                if (isEmpty) {
                    // Collapse this column
                    const btn = document.querySelector(`.column-toggle[data-col="${col}"]`);
                    if (btn && !btn.classList.contains('hidden')) {
                        btn.click();
                    }
                }
            });
        }

        // Run on page load
        checkEmptyColumns();

        // Error data from LLM judge review
        const judgeErrors = window.PACKET_CONFIG.judge_errors || [];

        // Mark rows that have errors
        function markErrorRows() {
            judgeErrors.forEach(err => {
                document.querySelectorAll('tr[data-tick-start]').forEach(row => {
                    const rowStart = parseInt(row.dataset.tickStart);
                    const rowEnd = parseInt(row.dataset.tickEnd);

                    // Check if this row overlaps with the error tick range
                    if (rowStart <= err.tick_end && rowEnd >= err.tick_start) {
                        row.classList.add('has-error');
                        if (err.source === 'agent') {
                            row.classList.add('has-agent-error');
                        } else {
                            row.classList.add('has-user-error');
                        }

                        // Update marker with error info
                        const marker = row.querySelector('.error-marker');
                        const tooltip = row.querySelector('.error-tooltip');
                        if (marker && tooltip) {
                            // Store error IDs for navigation
                            const existingIds = marker.dataset.errorIds || '';
                            marker.dataset.errorIds = existingIds ? `${existingIds},${err.id}` : err.id;

                            // Build tooltip content with DOM APIs — PACKET_CONFIG
                            // values (source/severity/tags) must never be
                            // interpolated into an HTML string (XSS sink).
                            const errorLine = document.createElement('div');
                            const sourceEl = document.createElement('strong');
                            sourceEl.textContent = String(err.source).toUpperCase();
                            errorLine.appendChild(sourceEl);
                            errorLine.appendChild(document.createTextNode(
                                ` (${err.severity}): ${(err.tags || []).join(', ')}`
                            ));
                            tooltip.appendChild(errorLine);
                        }
                    }
                });
            });
        }
        markErrorRows();

        // Click on error marker to navigate to error in review section
        document.querySelectorAll('.error-marker').forEach(marker => {
            marker.addEventListener('click', (e) => {
                e.stopPropagation();
                const errorIds = marker.dataset.errorIds;
                if (errorIds) {
                    const firstErrorId = errorIds.split(',')[0];
                    const errorItem = document.querySelector(`.error-item[data-error-id="${firstErrorId}"]`);
                    if (errorItem) {
                        // Open the review section if closed
                        const reviewSection = document.querySelector('.review-section details');
                        if (reviewSection && !reviewSection.open) {
                            reviewSection.open = true;
                        }

                        // Open this specific error
                        if (!errorItem.open) {
                            errorItem.open = true;
                        }

                        // Scroll to error
                        errorItem.scrollIntoView({ behavior: 'smooth', block: 'center' });

                        // Highlight animation
                        errorItem.style.transition = 'background 0.3s';
                        errorItem.style.background = '#ffeb3b';
                        setTimeout(() => {
                            errorItem.style.background = '';
                        }, 1500);
                    }
                }
            });
        });

        // Click on error ticks to navigate to that part of the conversation
        document.querySelectorAll('.clickable-error').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();  // Don't toggle the details
                const tickStart = parseInt(el.dataset.tickStart);
                const tickEnd = parseInt(el.dataset.tickEnd);

                // Find the first row that overlaps with this tick range
                let targetRow = null;
                document.querySelectorAll('tr[data-tick-start]').forEach(row => {
                    const rowStart = parseInt(row.dataset.tickStart);
                    const rowEnd = parseInt(row.dataset.tickEnd);

                    if (rowStart <= tickEnd && rowEnd >= tickStart && !targetRow) {
                        targetRow = row;
                    }
                });

                if (targetRow) {
                    // Scroll to row
                    targetRow.scrollIntoView({ behavior: 'smooth', block: 'center' });

                    // Add highlight animation
                    targetRow.classList.remove('error-highlight');
                    void targetRow.offsetWidth;  // Trigger reflow
                    targetRow.classList.add('error-highlight');

                    // Play audio from this point
                    const clickAudio = document.getElementById('mainAudio');
                    if (clickAudio) {
                        const startTime = parseFloat(targetRow.dataset.startTime);
                        if (!isNaN(startTime)) {
                            clickAudio.currentTime = startTime;
                            clickAudio.play();
                        }
                    }
                }
            });
        });


        // ============================================================
        // Audio controls (after sticky player is in DOM)
        // ============================================================

        const mainAudio = document.getElementById('mainAudio');
        if (mainAudio) {
            // Click on tick or time to seek
            document.querySelectorAll('.clickable-time').forEach(cell => {
                cell.addEventListener('click', (e) => {
                    const row = cell.closest('tr');
                    const startTime = parseFloat(row.dataset.startTime);
                    if (!isNaN(startTime)) {
                        mainAudio.currentTime = startTime;
                        mainAudio.play();

                        // Highlight current row
                        document.querySelectorAll('tr.playing').forEach(r => r.classList.remove('playing'));
                        row.classList.add('playing');
                    }
                });
            });

            // Update highlighted row during playback
            mainAudio.addEventListener('timeupdate', () => {
                const currentTime = mainAudio.currentTime;
                let activeRow = null;

                document.querySelectorAll('tr[data-start-time]').forEach(row => {
                    const startTime = parseFloat(row.dataset.startTime);
                    if (startTime <= currentTime) {
                        activeRow = row;
                    }
                });

                // Only update if changed (no auto-scroll so user can browse freely)
                const currentPlaying = document.querySelector('tr.playing');
                if (activeRow && activeRow !== currentPlaying) {
                    if (currentPlaying) currentPlaying.classList.remove('playing');
                    activeRow.classList.add('playing');
                }
            });

            // Clear highlight when audio ends
            mainAudio.addEventListener('ended', () => {
                document.querySelectorAll('tr.playing').forEach(r => r.classList.remove('playing'));
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                // Don't trigger if user is typing in an input
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                switch (e.code) {
                    case 'Space':
                        e.preventDefault();
                        if (mainAudio.paused) {
                            mainAudio.play();
                        } else {
                            mainAudio.pause();
                        }
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        mainAudio.currentTime = Math.max(0, mainAudio.currentTime - 5);
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        mainAudio.currentTime = Math.min(mainAudio.duration, mainAudio.currentTime + 5);
                        break;
                }
            });
        }
