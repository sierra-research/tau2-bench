# τ-bench Web Interface

![τ-bench Leaderboard](public/leaderboard.png)

## 🚀 Quick Start

### Prerequisites

- **Node.js** (version 16 or higher)
- **npm** (comes with Node.js)

### Installation & Setup

1. **Navigate to the leaderboard directory**
   ```bash
   cd web/leaderboard
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment (optional)**
   ```bash
   cp .env.example .env.local
   ```
   Uncomment `VITE_SUBMISSIONS_BASE_URL` to fetch submission and trajectory data from S3 instead of the local `public/` directory. This is required if trajectory files have been removed from the repo.

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   - Navigate to `http://localhost:5173` (or the URL shown in your terminal)
   - The application will automatically reload when you make changes

## Submitting to the Leaderboard

We welcome community submissions! The leaderboard accepts model evaluation results through pull requests for both **text** (standard) and **voice** (audio-native) modalities.

See the **[Leaderboard Submission Guide](../../docs/leaderboard-submission.md)** for complete instructions on running evaluations, preparing submissions, and submitting a pull request.

## 🔧 Development

### Project Structure
```
src/
├── components/          # React components
│   ├── DocsContent.jsx     # Documentation content display
│   ├── DocsContent.css     # Documentation styling
│   ├── Leaderboard.jsx     # Model performance leaderboard
│   ├── Leaderboard.css     # Leaderboard styling
│   ├── Results.jsx         # Results dashboard
│   ├── Results.css         # Results styling
│   ├── TrajectoryVisualizer.jsx  # Trajectory exploration
│   └── TrajectoryVisualizer.css  # Trajectory visualizer styling
├── assets/             # Static assets and data
│   ├── data/              # Research data and benchmark results
│   ├── arXiv-2506.07982v1/  # Paper content and figures
│   └── *.png, *.svg       # Logo images and icons
├── App.jsx             # Main application component
├── App.css             # Main application styling
├── index.css           # Global styles
└── main.jsx           # Application entry point

public/
├── submissions/        # Submission metadata (trajectories on S3)
├── task-data/          # Domain-specific tasks and policies
├── blog/               # Blog content
└── *.png, *.svg        # Public assets
```
