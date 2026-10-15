// Blog posts and author profiles for the Blog page and author bio pages.
// Bios, roles, and photos come from the authors' sierra.ai/author pages;
// photos are served locally from public/authors/<slug>.jpg.

export const PAPERS = {
  seataubench: {
    title: 'SEATauBench: Progressively Adapting Tool-Agent-User Evaluation Into Low-Resource Southeast Asian Languages',
    href: 'https://arxiv.org/abs/2606.28715v2',
    venue: 'Findings of EMNLP 2026',
  },
  tauBench: {
    title: 'τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains',
    href: 'https://arxiv.org/abs/2406.12045',
    venue: 'arXiv 2024',
  },
  tau2Bench: {
    title: 'τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment',
    href: 'https://arxiv.org/abs/2506.07982',
    venue: 'arXiv 2025',
  },
  tauKnowledge: {
    title: 'τ-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge',
    href: 'https://arxiv.org/abs/2603.04370',
    venue: 'arXiv 2026',
  },
  tauVoice: {
    title: 'τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains',
    href: 'https://arxiv.org/abs/2603.13686',
    venue: 'arXiv 2026',
  },
  tauRec: {
    title: 'τ-Rec: A Verifiable Benchmark for Agentic Recommender Systems',
    href: 'https://doi.org/10.1145/3773078.3831847',
    venue: 'ACM RecSys 2026',
  },
}

export const AUTHORS = {
  'my-chiffon-nguyen': {
    name: 'My Chiffon Nguyen',
    role: 'AI Researcher at SEACrowd',
    photo: '/community/assets/seataubench/my-chiffon-nguyen.jpg',
    bio: 'My (Chiffon) Nguyen is an AI researcher at SEACrowd working on AI safety and governance, multilingual and multicultural AI, and cooperative agents. A Fall 2026 SPAR Fellow, she has worked on AI policy simulation and multilingual agent evaluation, including SEATauBench.',
    bioSegments: [
      {
        text: 'My (Chiffon) Nguyen',
        href: 'https://mychiffonn.com/',
      },
      {
        text: ' is an AI researcher at SEACrowd working on ',
      },
      {
        text: 'AI safety and governance',
        strong: true,
      },
      {
        text: ', multilingual and multicultural AI, and cooperative agents. A Fall 2026 SPAR Fellow, she has worked on AI policy simulation and multilingual agent evaluation, including SEATauBench.',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'aulia-adila': {
    name: 'Aulia Adila',
    role: 'AI Engineer at AI Singapore',
    photo: '/community/assets/seataubench/aulia-adila.jpg',
    bio: 'Aulia Adila is an AI Engineer at AI Singapore. Her research spans speech processing and privacy, deepfake detection, and multilingual agent evaluation. She holds a master\'s degree in Information Science from JAIST and a bachelor\'s degree in Computer Science from Institut Teknologi Bandung.',
    bioSegments: [
      {
        text: 'Aulia Adila',
        href: 'https://www.linkedin.com/in/aulia-adila/?isSelfProfile=false',
      },
      {
        text: ' is an AI Engineer at AI Singapore. Her research spans ',
      },
      {
        text: 'speech processing and privacy',
        strong: true,
      },
      {
        text: ', deepfake detection, and multilingual agent evaluation. She holds a master\'s degree in Information Science from JAIST and a bachelor\'s degree in Computer Science from Institut Teknologi Bandung.',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'saksorn-ruangtanusak': {
    name: 'Saksorn Ruangtanusak',
    role: 'Researcher at SCBX Group',
    paperAffiliation: 'SEACrowd; SCB DataX, SCBX Group',
    photo: '/authors/saksorn-ruangtanusak.jpg',
    bio: 'Saksorn (Harry) Ruangtanusak is an AI researcher at SCBX Group, where he works with the Typhoon team on language models and AI agents. His research focuses on language agents, spanning multilingual agent evaluation (SEATauBench), behavioral fidelity in simulated personas (RoleCapBench), and efficient tool use (Talk Less, Call Right). He also collaborated with Professor Michael S. Bernstein on human behavioral simulation in “Can Vision-Language Models Simulate Human Behavioral Responses to Online Advertising Content?”',
    bioSegments: [
      {
        text: 'Saksorn (Harry) Ruangtanusak',
        href: 'https://saksornr.github.io/',
      },
      {
        text: ' is an AI researcher at SCBX Group, where he works with the Typhoon team on language models and AI agents. His research focuses on ',
      },
      {
        text: 'language agents',
        strong: true,
      },
      {
        text: ', spanning multilingual agent evaluation (SEATauBench), behavioral fidelity in simulated personas (RoleCapBench), and efficient tool use (Talk Less, Call Right). He also collaborated with Professor Michael S. Bernstein on human behavioral simulation in “Can Vision-Language Models Simulate Human Behavioral Responses to Online Advertising Content?”',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'kittiphat-leesombatwathana': {
    name: 'Kittiphat Leesombatwathana',
    role: 'Student at VISTEC',
    photo: '/community/assets/seataubench/kittiphat-leesombatwathana.jpg',
    bio: 'Kittiphat Leesombatwathana is a student at VISTEC whose research spans multilingual natural language processing, promise verification, and agent evaluation. He previously studied Computer Science at Chulalongkorn University and co-authored a Best System Paper at SemEval 2025.',
    bioSegments: [
      {
        text: 'Kittiphat Leesombatwathana',
        href: 'https://www.linkedin.com/in/kittiphat-leesombatwathana-a05170256/?isSelfProfile=false',
      },
      {
        text: ' is a student at VISTEC whose research spans ',
      },
      {
        text: 'multilingual natural language processing',
        strong: true,
      },
      {
        text: ', promise verification, and agent evaluation. He previously studied Computer Science at Chulalongkorn University and co-authored a Best System Paper at SemEval 2025.',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'vissuta-gunawan-lim': {
    name: 'Vissuta Gunawan Lim',
    role: 'Computer Science Undergraduate at University of Indonesia',
    photo: '/community/assets/seataubench/vissuta-gunawan-lim.jpg',
    bio: 'Vissuta Gunawan Lim is a Computer Science undergraduate at the University of Indonesia. His research interests span multilingual agent evaluation, efficient AI agents, and medical computer vision. He contributes to Southeast Asian LLM benchmarking through SEACrowd and to radiology report generation research at MLCV Lab.',
    bioSegments: [
      {
        text: 'Vissuta Gunawan Lim',
        href: 'https://vissutalim.com/',
      },
      {
        text: ' is a Computer Science undergraduate at the University of Indonesia. His research interests span ',
      },
      {
        text: 'multilingual agent evaluation',
        strong: true,
      },
      {
        text: ', efficient AI agents, and medical computer vision. He contributes to Southeast Asian LLM benchmarking through SEACrowd and to radiology report generation research at MLCV Lab.',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'patomporn-payoungkhamdee': {
    name: 'Patomporn Payoungkhamdee',
    role: 'PhD Student at VISTEC',
    photo: '/community/assets/seataubench/patomporn-payoungkhamdee.jpg',
    bio: 'Patomporn Payoungkhamdee is a PhD student at VISTEC\'s Natural Language and Representation Lab. His research focuses on multilingual distillation, tool-integrated reasoning, and cross-lingual AI, building on his contributions to Thai language models and benchmarks. He also mentors researchers through the SEACrowd Apprentice Program.',
    bioSegments: [
      {
        text: 'Patomporn Payoungkhamdee',
        href: 'https://calzonelover.github.io/',
      },
      {
        text: ' is a PhD student at VISTEC\'s Natural Language and Representation Lab. His research focuses on ',
      },
      {
        text: 'multilingual distillation',
        strong: true,
      },
      {
        text: ', tool-integrated reasoning, and cross-lingual AI, building on his contributions to Thai language models and benchmarks. He also mentors researchers through the SEACrowd Apprentice Program.',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'samuel-cahyawijaya': {
    name: 'Samuel Cahyawijaya',
    role: 'Member of Technical Staff at Cohere',
    photo: '/community/assets/seataubench/samuel-cahyawijaya.jpg',
    bio: 'Samuel Cahyawijaya is a Member of Technical Staff at Cohere and a research leader in Southeast Asian multilingual NLP. His work spans multilingual language modeling, resources for low-resource languages, and culturally grounded AI evaluation. He also mentors researchers through SEACrowd\'s apprenticeship program.',
    bioSegments: [
      {
        text: 'Samuel Cahyawijaya',
        href: 'https://samuelcahyawijaya.github.io/',
      },
      {
        text: ' is a Member of Technical Staff at Cohere and a research leader in ',
      },
      {
        text: 'Southeast Asian multilingual NLP',
        strong: true,
      },
      {
        text: '. His work spans multilingual language modeling, resources for low-resource languages, and culturally grounded AI evaluation. He also mentors researchers through SEACrowd\'s apprenticeship program.',
      },
    ],
    paperKeys: ['seataubench'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'bharath-narasimhan': {
    name: 'Bharath Sivaram Narasimhan',
    role: 'Independent Researcher',
    photo: '/community/assets/bharath-narasimhan.jpg',
    bio: 'Bharath Sivaram Narasimhan is an independent researcher focused on verifiable evaluation methods for reliable agentic AI.',
    paperKeys: ['tauRec'],
    backHref: '/community',
    backLabel: 'Community spotlight',
  },
  'victor-barres': {
    name: 'Victor Barres',
    role: 'Research Scientist at Sierra',
    sierraProfile: 'https://sierra.ai/author/victor-barres',
    bio: 'Victor Barres is a Research Scientist at Sierra specializing in conversational AI, dialogue modeling, and the integration of large language models with reasoning systems. He holds a PhD in Computational Cognitive Neuroscience from the University of Southern California. Previously, Victor was a Senior Researcher at Elemental Cognition, where he developed architectures that combine language models and symbolic reasoning systems to enable complex scientific question answering. As a Principal NLP Scientist at Uniphore, he led the NLP research team and drove advances in dialogue systems and language models for large-scale real-world applications.',
    paperKeys: ['tau2Bench', 'tauKnowledge', 'tauVoice'],
  },
  'ben-shi': {
    name: 'Ben Shi',
    role: 'Research at Sierra',
    sierraProfile: 'https://sierra.ai/author/ben-shi',
    bio: 'Ben is on the Research team at Sierra. Previously, he was at Princeton University and Meta.',
    paperKeys: ['tauKnowledge'],
  },
  'ola-zytek': {
    name: 'Ola Zytek',
    role: 'Research Engineer at Sierra',
    sierraProfile: 'https://sierra.ai/author/ola-zytek',
    bio: 'Ola Zytek is a Research Engineer at Sierra, focused on building conversational agents and retrieval systems through post training and evaluation. Prior to joining Sierra, she completed her PhD at MIT, where she specialized in building systems to support human decision-making with machine learning.',
    paperKeys: ['tauKnowledge'],
  },
  'soham-ray': {
    name: 'Soham Ray',
    role: 'Research Engineer at Sierra',
    sierraProfile: 'https://sierra.ai/author/soham-ray',
    bio: "Soham Ray is a Research Engineer at Sierra, focused on building conversational AI agents. He has over four years of experience in the customer service AI space, conducting applied research on augmenting and automating task-oriented dialogue systems. He holds a Master's degree in Computer Science from Cornell University, where he specialized in AI and NLP.",
    paperKeys: ['tauVoice', 'tau2Bench'],
  },
  'keshav-dhandhania': {
    name: 'Keshav Dhandhania',
    role: 'Research Engineer at Sierra',
    sierraProfile: 'https://sierra.ai/author/keshav-dhandhania',
    bio: "Keshav Dhandhania is a Research Engineer at Sierra, where he works on improving AI agents through evaluation frameworks, post-training, and actionable insights. Previously, he spent five years at Google on teams including Gemini @ Chrome, Gemini, and Google Assistant. Before that, he co-founded CommonLounge—later acquired by Brex—and earned his Master's degree from MIT specializing in deep learning and NLP.",
    paperKeys: ['tauVoice'],
  },
  'pedram-razavi': {
    name: 'Pedram Razavi',
    role: 'Software Engineer at Sierra',
    sierraProfile: 'https://sierra.ai/author/pedram-razavi',
    bio: 'Pedram Razavi is a software engineer on the Knowledge team at Sierra. Previously, he was an engineer at Cocoon and Quip. He studied Computer Science and Mathematics at MIT and earned an M.S. in Symbolic Systems from Stanford.',
    paperKeys: ['tauBench', 'tauKnowledge'],
  },
  'karthik-narasimhan': {
    name: 'Karthik Narasimhan',
    role: 'Head of Research at Sierra',
    sierraProfile: 'https://sierra.ai/author/karthik-narasimhan',
    bio: 'Karthik is Head of Research at Sierra and an associate professor of Computer Science at Princeton. He holds a PhD from MIT and co-authored the first GPT paper at OpenAI. He has been researching natural language processing, reinforcement learning, and AI agents for over a decade and has co-authored works like ReAct, Tree of Thoughts, CoALA and SWE-agent.',
    paperKeys: ['tauBench', 'tau2Bench', 'tauKnowledge', 'tauVoice', 'tauRec'],
  },
  'shunyu-yao': {
    name: 'Shunyu Yao',
    role: 'AI Researcher',
    // No sierra.ai author page; bio compiled from his public profile
    // (ysymyth.github.io) and the τ-bench paper.
    bio: 'Shunyu Yao is an AI researcher known for foundational work on language agents, including ReAct, Tree of Thoughts, Reflexion, and CoALA. He is the lead author of the original τ-bench. He holds a PhD in Computer Science from Princeton University, where he was a member of the Princeton NLP group, and studied in the Yao Class at Tsinghua University.',
    paperKeys: ['tauBench'],
  },
  'noah-shinn': {
    name: 'Noah Shinn',
    role: 'Research Scientist at Sierra',
    sierraProfile: 'https://sierra.ai/author/noah-shinn',
    bio: 'Noah is a research scientist at Sierra. Prior to Sierra, he worked on machine learning and programming language research at Northeastern and MIT.',
    paperKeys: ['tauBench'],
  },
  'honghua-dong': {
    name: 'Honghua Dong',
    role: 'PhD candidate, University of Toronto',
    sierraProfile: 'https://sierra.ai/author/honghua-dong',
    bio: 'Honghua Dong is a Ph.D. candidate at the University of Toronto. His research focuses on the development and evaluation of language model agents. He is currently interning at Sierra, where he is working to improve language model agents in the domain of customer service.',
    paperKeys: ['tau2Bench'],
  },
}

// Newest first. `href` values starting with http are external; everything else
// is resolved against import.meta.env.BASE_URL.
export const BLOG_POSTS = [
  {
    slug: 'stanford-aims-talk',
    title: 'The τ-bench Family: Guest Lecture at the Stanford AIMS Seminar',
    category: 'Talk',
    date: 'July 2026',
    description:
      'Slides from a guest lecture at the Stanford AIMS Seminar tracing the τ-bench family — from the original tool-agent-user substrate through dual control and knowledge to full-duplex voice — including how pausing time makes a realistic, controllable voice benchmark possible.',
    href: 'talks/stanford-aims-tau.html',
    authorSlugs: ['soham-ray', 'victor-barres', 'honghua-dong'],
  },
  {
    slug: 'tau-voice-sierra',
    title: 'τ-voice: Benchmarking Real-Time Voice Agents on Real-World Tasks',
    category: 'Research',
    date: 'May 2026',
    description:
      'τ-voice evaluates real-time voice agents on 278 grounded customer-service tasks, pairing deterministic task scoring with realistic, controllable audio — diverse personas, environmental noise, and free-form turn-taking.',
    href: 'https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks',
    authorSlugs: ['soham-ray', 'keshav-dhandhania', 'victor-barres'],
  },
  {
    slug: 'tau3-bench-announcement',
    title: 'τ³-Bench: Advancing Agent Benchmarking to Knowledge and Voice',
    category: 'Announcement',
    date: 'March 2026',
    description:
      'τ³-bench extends the benchmark family with the τ-knowledge and τ-voice tracks, advancing agent evaluation to unstructured knowledge and real-time voice.',
    href: 'https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice',
    authorSlugs: ['victor-barres', 'ben-shi', 'ola-zytek', 'soham-ray', 'keshav-dhandhania', 'pedram-razavi'],
  },
  {
    slug: 'tau-knowledge',
    title: 'τ-knowledge',
    category: 'Research',
    date: 'February 2026',
    description:
      'A benchmark for evaluating AI agents in knowledge-intensive customer support: a realistic fintech knowledge base of 698 documents paired with tasks requiring multi-step reasoning, policy application, and tool use.',
    href: 'blog/tau-knowledge.html',
    authorSlugs: ['ben-shi', 'ola-zytek', 'pedram-razavi'],
  },
  {
    slug: 'tau-voice-examples',
    title: 'τ-voice Examples',
    category: 'Research',
    date: 'February 2026',
    description:
      'Annotated examples of τ-voice calls — overlapping speech, interruptions, accents, and background noise — showing how the same task can succeed with clean audio and fail under realistic conditions.',
    href: 'blog/tau-voice-examples.html',
    authorSlugs: ['soham-ray', 'keshav-dhandhania', 'victor-barres'],
  },
  {
    slug: 'tau3-task-fixes',
    title: 'τ³-Bench: Fixing Airline + Retail',
    category: 'Engineering',
    date: 'February 2026',
    description:
      'We audited and fixed 50+ tasks across the airline and retail domains, addressing incorrect expected actions, ambiguous instructions, impossible constraints, and missing fallback behaviors.',
    href: 'blog/tau3-task-fixes.html',
    authorSlugs: ['victor-barres', 'ben-shi'],
  },
  {
    slug: 'tau2-bench-announcement',
    title: 'τ²-bench: Benchmarking Agents in Collaborative Real-World Scenarios',
    category: 'Announcement',
    date: 'June 2025',
    description:
      'τ²-bench evaluates conversational agents in dual-control environments, where both the agent and the user can act on the world.',
    href: 'https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios',
    authorSlugs: ['victor-barres', 'honghua-dong', 'soham-ray', 'karthik-narasimhan'],
  },
  {
    slug: 'tau-bench-announcement',
    title: 'τ-bench: Benchmarking AI Agents',
    category: 'Announcement',
    date: 'June 2024',
    description:
      'The original τ-bench, a benchmark for tool-agent-user interaction in real-world domains.',
    href: 'https://sierra.ai/blog/benchmarking-ai-agents',
    authorSlugs: ['shunyu-yao', 'noah-shinn', 'pedram-razavi', 'karthik-narasimhan'],
  },
]

export const postsByAuthor = (slug) => BLOG_POSTS.filter((p) => p.authorSlugs.includes(slug))

export const authorPhoto = (slug) =>
  AUTHORS[slug].photo ?? `${import.meta.env.BASE_URL}authors/${slug}.jpg`
