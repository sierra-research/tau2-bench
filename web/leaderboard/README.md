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

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   - Navigate to `http://localhost:5173` (or the URL shown in your terminal)
   - The application will automatically reload when you make changes

## 📊 Submitting to the Leaderboard

We welcome community submissions! The leaderboard accepts model evaluation results through pull requests for both **text** (standard) and **voice** (audio-native) modalities.

### How to Submit

1. **Evaluate your model** using [tau-bench](https://github.com/sierra-research/tau2-bench)
2. **Create a JSON submission** following our schema (see `public/submissions/schema.json`)
3. **Submit a pull request** with your results file and trajectory links for verification

### Quick Example (Text)

```json
{
  "model_name": "My-Model-v1.0",
  "model_organization": "My Organization",
  "submitting_organization": "My Organization",
  "submission_date": "2025-01-15",
  "contact_info": {
    "email": "contact@myorg.com",
    "name": "Research Team"
  },
  "trajectories_available": true,
  "results": {
    "retail": {"pass_1": 75.2, "pass_2": 68.1, "pass_3": null, "pass_4": null},
    "airline": {"pass_1": 61.2, "pass_2": null, "pass_3": null, "pass_4": null},
    "telecom": {"pass_1": 45.6, "pass_2": null, "pass_3": null, "pass_4": null},
    "banking_knowledge": {"pass_1": 22.5, "pass_2": 17.3, "pass_3": null, "pass_4": null, "retrieval_config": "terminal"}
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "v1.0",
    "user_simulator": "gpt-4.1-2025-04-14",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": true,
      "details": "Only evaluated Pass@1 for all domains"
    }
  }
}
```

### Quick Example (Voice)

```json
{
  "model_name": "My-Voice-Model",
  "model_organization": "My Organization",
  "submitting_organization": "My Organization",
  "submission_date": "2026-03-11",
  "modality": "voice",
  "contact_info": {
    "email": "contact@myorg.com"
  },
  "trajectories_available": false,
  "results": {
    "retail": {"pass_1": 43.9},
    "airline": {"pass_1": 40.0},
    "telecom": {"pass_1": 21.1}
  },
  "voice_config": {
    "provider": "openai",
    "model": "gpt-realtime-1.5",
    "tick_duration_seconds": 0.2,
    "max_steps_seconds": 600,
    "user_tts_provider": "elevenlabs/eleven_v3"
  },
  "methodology": {
    "evaluation_date": "2026-03-01",
    "user_simulator": "gpt-4.1-2025-04-14",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false
    }
  }
}
```

### 🔍 Verification System

The leaderboard now includes a verification system to ensure result quality:

- **✅ Verified submissions** have trajectory data, use standard prompts, and complete all evaluations
- **⚠️ Unverified submissions** are marked with caution icons and may have missing data or modified methodologies
- Click on any model name to see detailed verification status and methodology information

### 📚 Model References

Each submission can include links to papers, documentation, and other resources about the model. This helps researchers access relevant information directly from the leaderboard. References are displayed in the model detail view with categorized badges for easy identification.

📋 **See `SUBMISSION_GUIDE.md` for complete submission instructions**

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
├── data/               # CSV files and cost information
├── task-data/          # Domain-specific tasks and policies
├── trajectory-data/    # Model execution trajectories
└── *.png, *.svg       # Public assets
```
