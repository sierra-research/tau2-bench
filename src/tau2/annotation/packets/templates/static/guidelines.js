        // Guidelines modal
        const modal = document.getElementById('guidelinesModal');
        const helpBtn = document.getElementById('helpBtn');
        const modalClose = document.getElementById('modalClose');

        function openGuidelines() {
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeGuidelines() {
            modal.classList.remove('active');
            document.body.style.overflow = '';
        }

        helpBtn.addEventListener('click', openGuidelines);
        modalClose.addEventListener('click', closeGuidelines);

        modal.addEventListener('click', (e) => {
            if (e.target === modal) { closeGuidelines(); }
        });

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            if (e.key === 'g' || e.key === 'G') {
                if (!modal.classList.contains('active')) {
                    e.preventDefault();
                    openGuidelines();
                }
            }
            if (e.key === 'Escape' && modal.classList.contains('active')) {
                closeGuidelines();
            }
        });
