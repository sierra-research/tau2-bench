// Generates static, indexable author bio pages (public/authors/<slug>.html)
// plus sitemap.xml and robots.txt from src/data/blogData.js.
//
// Runs automatically as part of `npm run build`; run `npm run generate` after
// editing blogData.js to refresh the committed pages.

import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { AUTHORS, PAPERS, BLOG_POSTS } from '../src/data/blogData.js'

const SITE = 'https://taubench.com'
const publicDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public')

const esc = (s) =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')

const postUrl = (post) => (post.href.startsWith('http') ? post.href : `/${post.href}`)

const truncate = (s, n) => (s.length <= n ? s : `${s.slice(0, s.lastIndexOf(' ', n))}…`)

const authorLine = (slugs) => {
  const links = slugs.map(
    (slug) => `<a href="/authors/${slug}.html" class="post-author-name">${esc(AUTHORS[slug].name)}</a>`
  )
  const names = links.length > 1 ? `${links.slice(0, -1).join(', ')} &amp; ${links.at(-1)}` : links[0]
  const photos = slugs
    .map(
      (slug) =>
        `<a href="/authors/${slug}.html" class="post-author-photo-link" title="${esc(AUTHORS[slug].name)}"><img src="/authors/${slug}.jpg" alt="${esc(AUTHORS[slug].name)}" class="post-author-photo" /></a>`
    )
    .join('')
  return `<div class="post-authors"><div class="post-author-photos">${photos}</div><span class="post-author-names">${names}</span></div>`
}

const postCard = (post) => {
  const external = post.href.startsWith('http')
  return `<article class="blog-card">
        <a class="blog-card-link" href="${postUrl(post)}"${external ? ' target="_blank" rel="noopener noreferrer"' : ''}>
          <div class="blog-card-top">
            <span class="blog-card-label">${esc(post.category)} · ${esc(post.date)}</span>
            ${external ? '<span class="blog-card-label">sierra.ai ↗</span>' : ''}
          </div>
          <h2 class="blog-card-title">${esc(post.title)}</h2>
          <p class="blog-card-description">${esc(post.description)}</p>
        </a>
        ${authorLine(post.authorSlugs)}
      </article>`
}

const STYLE = `
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
      -webkit-font-smoothing: antialiased;
      color: #2d3748;
      line-height: 1.6;
      background: #ffffff;
    }
    .navbar {
      position: sticky; top: 0; z-index: 1000;
      background: rgba(255, 255, 255, 0.98);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid #e2e8f0;
      padding: 16px 0;
    }
    .nav-container {
      max-width: 1100px; margin: 0 auto; padding: 0 24px;
      display: flex; align-items: center; justify-content: space-between;
    }
    .nav-logo { display: flex; flex-direction: column; gap: 2px; }
    .logo-main { text-decoration: none; font-size: 24px; font-weight: 800; }
    .tau-symbol { color: #065f46; }
    .bench-text { color: #2d3748; }
    .logo-attribution { display: flex; align-items: center; gap: 6px; text-decoration: none; }
    .sierra-logo { width: 14px; height: 14px; border-radius: 3px; }
    .from-text { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .nav-links { display: flex; align-items: center; gap: 28px; }
    .nav-links a { color: #718096; text-decoration: none; font-weight: 500; font-size: 15px; }
    .nav-links a:hover { color: #065f46; }
    .page {
      max-width: 800px; margin: 0 auto; padding: 48px 24px 80px;
    }
    .back-link {
      display: inline-block; color: #065f46; text-decoration: none;
      font-size: 14px; font-weight: 500; margin-bottom: 32px;
    }
    .back-link:hover { text-decoration: underline; }
    .author-header { display: flex; align-items: center; gap: 20px; margin-bottom: 20px; }
    .author-photo {
      width: 80px; height: 80px; border-radius: 50%;
      object-fit: cover; border: 1px solid #e2e8f0;
    }
    .author-name { font-size: 2rem; font-weight: 800; color: #1a202c; line-height: 1.15; }
    .author-role { font-size: 15px; color: #718096; margin-top: 4px; }
    .author-bio { font-size: 15px; line-height: 1.7; color: #4a5568; margin-bottom: 40px; }
    .section { margin-bottom: 40px; }
    .section-title {
      font-size: 11px; font-weight: 600; color: #a0aec0;
      text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px;
    }
    .paper-list { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    .paper-link { color: #2d3748; text-decoration: none; font-size: 15px; font-weight: 500; }
    .paper-link:hover { color: #065f46; text-decoration: underline; }
    .paper-venue { color: #a0aec0; font-size: 13px; margin-left: 8px; white-space: nowrap; }
    .blog-grid { display: flex; flex-direction: column; gap: 16px; }
    .blog-card {
      background: white; border: 1px solid #e2e8f0; border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04); padding: 24px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .blog-card:hover { border-color: #cbd5e0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06); }
    .blog-card-link { display: block; text-decoration: none; color: inherit; }
    .blog-card-top {
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; margin-bottom: 10px;
    }
    .blog-card-label {
      font-size: 11px; font-weight: 600; color: #a0aec0;
      text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;
    }
    .blog-card-title { font-size: 1.25rem; font-weight: 700; color: #1a202c; line-height: 1.3; margin-bottom: 8px; }
    .blog-card-link:hover .blog-card-title { color: #065f46; }
    .blog-card-description { font-size: 14px; line-height: 1.55; color: #4a5568; margin-bottom: 16px; }
    .post-authors { display: flex; align-items: center; gap: 10px; }
    .post-author-photos { display: flex; flex-shrink: 0; }
    .post-author-photo-link { display: block; border-radius: 50%; }
    .post-author-photo-link:not(:first-child) { margin-left: -8px; }
    .post-author-photo {
      display: block; width: 28px; height: 28px; border-radius: 50%;
      border: 2px solid white; object-fit: cover;
    }
    .post-author-names { font-size: 13px; color: #718096; line-height: 1.4; }
    .post-author-name { color: #4a5568; text-decoration: none; font-weight: 500; }
    .post-author-name:hover { color: #065f46; text-decoration: underline; }
    .footer {
      border-top: 1px solid #e2e8f0; padding: 24px; text-align: center;
      font-size: 14px; color: #718096; background: #f8fafc;
    }
    .footer a { color: #065f46; text-decoration: none; }
    @media (max-width: 640px) {
      .nav-links { gap: 16px; }
      .nav-links a { font-size: 13px; }
      .page { padding: 32px 16px 48px; }
      .author-name { font-size: 1.6rem; }
      .author-photo { width: 64px; height: 64px; }
      .blog-card { padding: 18px; }
    }
`

const NAV = `<nav class="navbar">
    <div class="nav-container">
      <div class="nav-logo">
        <a href="/" class="logo-main"><span class="tau-symbol">τ</span><span class="bench-text">-bench</span></a>
        <a href="https://sierra.ai" target="_blank" rel="noopener noreferrer" class="logo-attribution">
          <img src="/sierra_logo.jpeg" alt="Sierra" class="sierra-logo" />
          <span class="from-text">from Sierra</span>
        </a>
      </div>
      <div class="nav-links">
        <a href="/#home">Overview</a>
        <a href="/#leaderboard">Leaderboard</a>
        <a href="/#trajectory-visualizer">Visualizer</a>
        <a href="/#blog">Blog</a>
        <a href="https://github.com/sierra-research/tau2-bench" target="_blank" rel="noopener noreferrer">GitHub</a>
      </div>
    </div>
  </nav>`

const authorPage = (slug, author) => {
  const description = truncate(author.bio, 155)
  const posts = BLOG_POSTS.filter((p) => p.authorSlugs.includes(slug))
  const papers = author.paperKeys.map((key) => PAPERS[key])
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${esc(author.name)} | τ-bench</title>
  <meta name="description" content="${esc(description)}">
  <link rel="canonical" href="${SITE}/authors/${slug}.html">
  <meta property="og:type" content="profile">
  <meta property="og:title" content="${esc(author.name)} | τ-bench">
  <meta property="og:description" content="${esc(description)}">
  <meta property="og:image" content="${SITE}/authors/${slug}.jpg">
  <meta property="og:url" content="${SITE}/authors/${slug}.html">
  <meta name="twitter:card" content="summary">
  <style>${STYLE}</style>
</head>
<body>
  ${NAV}

  <div class="page">
    <a href="/#blog" class="back-link">← All posts</a>
    <header class="author-header">
      <img src="/authors/${slug}.jpg" alt="${esc(author.name)}" class="author-photo" />
      <div>
        <h1 class="author-name">${esc(author.name)}</h1>
        <p class="author-role">${esc(author.role)}</p>
      </div>
    </header>
    <p class="author-bio">${esc(author.bio)}</p>

    <section class="section">
      <h2 class="section-title">Papers</h2>
      <ul class="paper-list">
        ${papers
          .map(
            (paper) =>
              `<li><a href="${paper.href}" target="_blank" rel="noopener noreferrer" class="paper-link">${esc(paper.title)}</a><span class="paper-venue">${esc(paper.venue)}</span></li>`
          )
          .join('\n        ')}
      </ul>
    </section>

    ${
      posts.length
        ? `<section class="section">
      <h2 class="section-title">Posts</h2>
      <div class="blog-grid">
      ${posts.map(postCard).join('\n      ')}
      </div>
    </section>`
        : ''
    }
  </div>

  <footer class="footer">
    <p>For questions or feedback, contact <a href="mailto:research@sierra.ai">research@sierra.ai</a></p>
  </footer>
</body>
</html>
`
}

const authorsDir = join(publicDir, 'authors')
mkdirSync(authorsDir, { recursive: true })

for (const [slug, author] of Object.entries(AUTHORS)) {
  writeFileSync(join(authorsDir, `${slug}.html`), authorPage(slug, author))
}

const urls = [
  `${SITE}/`,
  ...BLOG_POSTS.filter((p) => !p.href.startsWith('http')).map((p) => `${SITE}/${p.href}`),
  ...Object.keys(AUTHORS).map((slug) => `${SITE}/authors/${slug}.html`),
]

writeFileSync(
  join(publicDir, 'sitemap.xml'),
  `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((url) => `  <url><loc>${url}</loc></url>`).join('\n')}
</urlset>
`
)

writeFileSync(
  join(publicDir, 'robots.txt'),
  `User-agent: *
Allow: /

Sitemap: ${SITE}/sitemap.xml
`
)

console.log(`Generated ${Object.keys(AUTHORS).length} author pages, sitemap.xml, robots.txt`)
