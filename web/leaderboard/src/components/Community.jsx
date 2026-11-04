import { useEffect } from 'react'
import './Blog.css'
import './Community.css'
import { COMMUNITY_EXTENSIONS } from '../data/communityExtensions'

const resolveHref = (href) =>
  href.startsWith('http') ? href : `${import.meta.env.BASE_URL}${href}`

function CommunityAuthorLine({ authors, displayAffiliations }) {
  const affiliations = displayAffiliations ?? authors.map((author) => author.affiliation)

  return (
    <div className="post-authors community-authors">
      <div className="post-author-photos">
        {authors.map((author) => {
          const external = author.profileUrl.startsWith('http')
          return (
            <a
              key={author.name}
              href={resolveHref(author.profileUrl)}
              className="post-author-photo-link"
              title={`${author.name} — ${author.affiliation}`}
              {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
            >
              <img src={resolveHref(author.photo)} alt={author.name} className="post-author-photo" />
            </a>
          )
        })}
      </div>
      <div className="community-author-details">
        <span className="post-author-names">
          {authors.map((author, index) => {
            const external = author.profileUrl.startsWith('http')
            return (
              <span key={author.name}>
                {index > 0 && (index === authors.length - 1 ? ' & ' : ', ')}
                <a
                  href={resolveHref(author.profileUrl)}
                  className="post-author-name"
                  {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
                >
                  {author.name}
                </a>
              </span>
            )
          })}
        </span>
        <span className="community-author-affiliations">
          {affiliations.join(' · ')}
        </span>
      </div>
    </div>
  )
}

function CommunityCard({ extension }) {
  return (
    <article className="blog-card community-card">
      <a
        className="blog-card-link"
        href={resolveHref(extension.href)}
      >
        <div className="blog-card-top">
          <span className="blog-card-label">{extension.category} · {extension.date}</span>
          <span className="blog-card-source">Read more →</span>
        </div>
        <h2 className="blog-card-title">{extension.title}</h2>
        <p className="blog-card-description">{extension.description}</p>
      </a>
      <CommunityAuthorLine authors={extension.authors} displayAffiliations={extension.displayAffiliations} />
    </article>
  )
}

function Community() {
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  return (
    <div className="blog-page community-page">
      <header className="blog-page-header">
        <h1 className="blog-page-title">Community Spotlight</h1>
        <p className="blog-page-subtitle">
          Projects from the research community that build on τ-bench and extend
          agent evaluation to new domains and settings.
        </p>
      </header>
      <div className="blog-grid">
        {COMMUNITY_EXTENSIONS.map((extension) => (
          <CommunityCard key={extension.slug} extension={extension} />
        ))}
      </div>
    </div>
  )
}

export default Community
