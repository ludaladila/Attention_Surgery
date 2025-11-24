# Attention Surgery: Interactive Mechanistic Interpretability for GPT-2

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)

**Attention Surgery** is an interactive visualization and analysis tool designed to perform "neurosurgery" on Large Language Models (specifically GPT-2 Small). It allows researchers and enthusiasts to ablate (disable) specific attention heads in real-time, observe the impact on generated text, and quantify the importance of different model components using advanced interpretability metrics.

---

## Core Concepts & Technology

This project implements several key techniques from the field of **Mechanistic Interpretability**:

### 1. Attention Head Ablation (The "Surgery")
We can selectively intervene on the model's internal computation by modifying the attention patterns of specific heads during inference.
- **Hard Zero Ablation**: Completely zeros out the attention pattern of a head, effectively removing it from the circuit.
- **Mean Ablation**: Replaces a head's attention pattern with a uniform distribution, preserving the magnitude of information flow but destroying its selectivity.
- **Previous Pattern**: Replaces a head's pattern with that of the same head from the previous layer, testing layer-wise dependencies.

### 2. Importance Scoring
We automatically identify "critical" heads using multiple scoring methods:
- **Gradient-based Importance**: Computes the gradient of the loss with respect to attention patterns to see which heads the model "cares" about most.
- **Rollback (L2) Importance**: Measures the L2 norm of the head's output vector, indicating the magnitude of its contribution to the residual stream.
- **Ablation Impact**: Measures the KL Divergence between the original logits and the logits after ablating a single head.

### 3. Logit Lens (Layer-wise Decoding)
We apply the model's "Unembedding" matrix to the residual stream at **intermediate layers**. This reveals what the model "believes" the next token is at each stage of processing, visualizing how confidence builds up layer by layer.

### 4. Metrics
- **KL Divergence**: Measures how much the output probability distribution changes after ablation.
- **Perplexity Delta**: Quantifies how much "confused" the model becomes (higher perplexity = worse prediction).
- **Top-1 Changed Ratio**: The percentage of tokens where the model's first-choice prediction changed.

---

##  Architecture

- **Backend (Python/FastAPI)**:
  - Uses `TransformerLens` (HookedTransformer) for model manipulation.
  - Implements PyTorch hooks to intercept and modify activations.
  - Exposes REST APIs for inference and metrics computation.

- **Frontend (React/Vite)**:
  - Built with React 18, Tailwind CSS, and Recharts.
  - Provides an interactive 12x12 Grid for head selection.
  - Visualizes token probabilities and attention impacts.
  
---

## How to Run Locally

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Backend Setup (FastAPI)

```bash
# Clone the repository
git clone https://github.com/ludaladila/Attention_Surge
cd attention_surgery

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn attention_surgery.api:app --reload --port 8000
```
The backend will start at `http://localhost:8000`.

### 2. Frontend Setup (React)

Open a new terminal window:

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The frontend will start at `http://localhost:5173`.

---

##  Deployment
[Web Link](https://attentionsurgery.onrender.com/)



---

##  How to Use

1. **Input Prompt**: Enter a starting text (e.g., "The Eiffel Tower is located in").
2. **Select Heads**: Click on the 12x12 grid to "ablate" (turn off) specific attention heads. Red cells indicate disabled heads.
3. **Auto-Select**: Use "Ablate Top-5 Heads" to automatically disable the most important heads based on gradient scores.
4. **Run Inference**: Click "Run Inference".
5. **Analyze Results**:
   - **Token Stream**: Compare the "Ablated" text vs. "Control" (original) text.
   - **Logit Lens**: Click on any generated token to see how the model's confidence evolved across layers.
   - **Metrics**: Check KL Divergence and Perplexity to gauge the damage done to the model.

---

##  License

MIT License. See [LICENSE](LICENSE) for details.
