// Blog posts and author profiles for the Blog page and author bio pages.
// Bios, roles, and photos come from the authors' sierra.ai/author pages;
// photos are served locally from public/authors/<slug>.jpg.

export const PAPERS = {
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
}

export const AUTHORS = {
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
    paperKeys: ['tauBench', 'tau2Bench', 'tauKnowledge', 'tauVoice'],
  },
  'shunyu-yao': {
    name: 'Shunyu Yao',
    role: 'AI Researcher',
    // No sierra.ai author page; bio compiled from his public profile
    // (ysymyth.github.io) and the τ-bench paper
    bio: 'Shunyu Yao is an AI researcher known for co-authoring ReAct, Tree of Thoughts, and τ-bench. He completed his PhD at Princeton University.',
    paperKeys: ['tauBench'],
  },
}

// Known task evaluation fixes and errata.
// Used by the leaderboard and visualizer to surface notes about specific tasks.
export const TASK_ERRATA = {
  'banking_knowledge/task_048': {
    issue: 463,
    summary:
      'task_048: all 24 gold actions performed, reward 0.00 — ambiguous user goal makes the simulator non-uniform, and loose write args penalize equivalent answers.',
    details:
      'The user goal in task_048 was underspecified, causing the simulator to behave non-uniformly across runs. Additionally, the reward function used strict argument matching for write-type tool calls, which incorrectly penalized semantically equivalent answers (e.g., different but valid field orderings or value representations). Fixed by: (1) clarifying the user goal to remove ambiguity, and (2) updating the reward checker to accept equivalent write arguments rather than requiring exact string matches.',
    fixedIn: 'banking_knowledge v1.1',
  },
}