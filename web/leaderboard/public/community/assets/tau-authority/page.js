// Follows assets/seataubench/page.js: the mobile navigation and anchors into
// disclosures. Every number on the page is already in the HTML, generated
// from the saved results by src/tau2/authority/spotlight.py, so nothing here
// fetches data.

const toggle = document.querySelector('.mobile-menu-toggle')
const nav = document.querySelector('#site-nav')
toggle.addEventListener('click', () => {
  const open = toggle.getAttribute('aria-expanded') !== 'true'
  toggle.setAttribute('aria-expanded', String(open))
  nav.classList.toggle('open', open)
})
nav.addEventListener('click', (e) => {
  if (e.target.closest('a')) {
    nav.classList.remove('open')
    toggle.setAttribute('aria-expanded', 'false')
  }
})
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && nav.classList.contains('open')) {
    nav.classList.remove('open')
    toggle.setAttribute('aria-expanded', 'false')
    toggle.focus()
  }
})
// Keep shared anchors useful when their content lives inside a disclosure.
function revealHashTarget() {
  const target = document.getElementById(location.hash.slice(1))
  if (!target) return
  let element = target
  while (element) {
    if (element.tagName === 'DETAILS') element.open = true
    element = element.parentElement
  }
}
revealHashTarget()
window.addEventListener('hashchange', revealHashTarget)
