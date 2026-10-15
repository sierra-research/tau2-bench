import { AUTHORS } from './blogData.js'

// Community projects that build on τ-bench. New entries should include a
// concise, factual summary and point to a first-party project page.
export const COMMUNITY_EXTENSIONS = [
  {
    slug: 'seataubench',
    title: 'SEATauBench',
    category: 'Research',
    date: 'October 2026',
    description: 'Extends τ²-bench to five target languages across Airline, Retail, and Telecom. Four localization settings separate dialogue, tool understanding, and domain reasoning. Published at EMNLP 2026.',
    displayAffiliations: [
      'SEACrowd',
      'AI Singapore',
      'SCB DataX, SCBX Group',
      'Chulalongkorn University',
      'VISTEC',
      'University of Indonesia',
      'Cohere',
    ],
    href: 'community/seataubench.html',
    githubUrl: 'https://github.com/SEACrowd/SEATauBench',
    paperUrl: 'https://arxiv.org/abs/2606.28715v2',
    leaderboardUrl: 'community/seataubench.html#leaderboard',
    authors: [
      'my-chiffon-nguyen',
      'aulia-adila',
      'saksorn-ruangtanusak',
      'kittiphat-leesombatwathana',
      'vissuta-gunawan-lim',
      'patomporn-payoungkhamdee',
      'samuel-cahyawijaya',
    ].map(slug => ({
      slug,
      name: AUTHORS[slug].name,
      affiliation: AUTHORS[slug].paperAffiliation ?? AUTHORS[slug].role,
      profileUrl: `authors/${slug}.html`,
      photo: AUTHORS[slug].photo.slice(1),
    })),
  },
  {
    slug: 'tau-rec',
    title: 'τ-Rec',
    category: 'Research',
    date: 'September 2026',
    description:
      'Extends τ-bench to conversational recommendation, evaluating whether agents can elicit user preferences, use catalog tools, follow policy, and make verifiably correct recommendations across multi-turn interactions. Published at ACM RecSys 2026.',
    href: 'community/tau-rec.html',
    githubUrl: 'https://github.com/nbharaths/tau-rec',
    paperUrl: 'https://doi.org/10.1145/3773078.3831847',
    leaderboardUrl: 'https://github.com/nbharaths/tau-rec/blob/main/LEADERBOARD.md',
    authors: [
      {
        slug: 'bharath-narasimhan',
        name: 'Bharath Sivaram Narasimhan',
        affiliation: 'Independent Researcher',
        profileUrl: 'authors/bharath-narasimhan.html',
        photo: 'community/assets/bharath-narasimhan.jpg',
      },
      {
        slug: 'karthik-narasimhan',
        name: 'Karthik Narasimhan',
        affiliation: 'Princeton University',
        profileUrl: 'authors/karthik-narasimhan.html',
        photo: 'authors/karthik-narasimhan.jpg',
      },
    ],
  },
]
