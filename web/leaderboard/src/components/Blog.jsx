import { useEffect } from 'react'
import './Blog.css'
import { BLOG_POSTS, AUTHORS, authorPhoto } from '../data/blogData'

const resolveHref = (href) =>
  href.startsWith('http') ? href : `${import.meta.env.BASE_URL}${href}`

const authorPageUrl = (slug) => `${import.meta.env.BASE_URL}authors/${slug}.html`

export function AuthorLine({ slugs }) {
  return (
    <div className="post-authors">
      <div className="post-author-photos">
        {slugs.map((slug) => (
          <a key={slug} href={authorPageUrl(slug)} className="post-author-photo-link" title={AUTHORS[slug]?.name}>
            <img src={authorPhoto(slug)} alt={AUTHORS[slug]?.name} className="post-author-photo" />
          </a>
        ))}
      </div>
      <span className="post-author-names">
        {slugs.map((slug, i) => (
          <span key={slug}>
            {i > 0 && (i === slugs.length - 1 ? ' & ' : ', ')}
            <a href={authorPageUrl(slug)} className="post-author-name">{AUTHORS[slug]?.name}</a>
          </span>
        ))}
      </span>
    </div>
  )
}

export function BlogCard({ post }) {
  const external = post.href.startsWith('http')
  return (
    <article className="blog-card">
      <a
        className="blog-card-link"
        href={resolveHref(post.href)}
        {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      >
        <div className="blog-card-top">
          <span className="blog-card-label">{post.category} · {post.date}</span>
          {external && <span className="blog-card-source">sierra.ai ↗</span>}
        </div>
        <h2 className="blog-card-title">{post.title}</h2>
        <p className="blog-card-description">{post.description}</p>
      </a>
      <AuthorLine slugs={post.authorSlugs} />
    </article>
  )
}

function Blog() {
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  return (
    <div className="blog-page">
      <header className="blog-page-header">
        <h1 className="blog-page-title">Blog</h1>
        <p className="blog-page-subtitle">
          Research updates, benchmark releases, and engineering notes from the τ-bench team.
        </p>
      </header>
      <div className="blog-grid">
        {BLOG_POSTS.map((post) => (
          <BlogCard key={post.slug} post={post} />
        ))}
      </div>
    </div>
  )
}

export default Blog
