<script setup>
import { ref } from 'vue'

const sections = [
  { id: 'getting-started', label: 'Getting Started' },
  { id: 'metrics', label: 'Metrics View' },
  { id: 'decisions', label: 'Decisions View' },
  { id: 'hyperparams', label: 'Hyperparams View' },
  { id: 'sessions', label: 'Sessions View' },
  { id: 'embeddings', label: 'Embeddings View' },
  { id: 'predictions', label: 'Predictions View' },
  { id: 'weights', label: 'Weights View' },
  { id: 'glossary', label: 'Glossary' },
]

const active = ref('getting-started')
function scrollTo(id) {
  active.value = id
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
}
</script>
<template>
  <div class="view">
    <div class="view-header"><h2>Help &amp; Guide</h2></div>

    <nav class="help-toc">
      <a v-for="s in sections" :key="s.id" :class="{ active: active === s.id }" @click="scrollTo(s.id)">{{ s.label }}</a>
    </nav>

    <!-- Getting Started -->
    <section :id="'getting-started'" class="help-section">
      <h3>Getting Started</h3>
      <p>RASHKOGIE is an end-to-end pipeline that learns to play GTA V by watching you play. Here's how the full loop works:</p>
      <ol>
        <li><strong>Capture</strong> — Play GTA V with the capture module running. It records screen frames, controller inputs, and game state at each timestep.</li>
        <li><strong>Tokenize</strong> — Raw frames and inputs are converted into compact token sequences that the model can learn from.</li>
        <li><strong>Train</strong> — The model trains on tokenized episodes. Each training stage (driving, combat, navigation, strategy) focuses on a different skill.</li>
        <li><strong>Monitor</strong> — This dashboard shows live metrics from the training loop via WebSocket. You can watch loss converge, inspect decisions, tune hyperparameters, and manage checkpoints.</li>
      </ol>
      <p>To get started: run a capture session, then launch training with <code>python -m pipeline.train</code>, and open this dashboard to watch progress in real time.</p>
    </section>

    <!-- Metrics View -->
    <section :id="'metrics'" class="help-section">
      <h3>Metrics View</h3>
      <p>Shows the core training signals over time for each module (driving, combat, navigation, strategy).</p>
      <ul>
        <li><strong>Loss curve</strong> — Should trend downward. Spikes are normal after learning-rate changes or new data batches. Sustained increases indicate overfitting or data issues.</li>
        <li><strong>Reward curve</strong> — Per-step reward signal. Higher is better. Look for steady upward trends with decreasing variance.</li>
        <li><strong>Episode Return</strong> — Total reward summed over an episode. This is the primary success metric. Plateaus may indicate the model needs new data or hyperparameter tuning.</li>
      </ul>
      <p>Use the module selector in the header to switch between training stages.</p>
    </section>

    <!-- Decisions View -->
    <section :id="'decisions'" class="help-section">
      <h3>Decisions View</h3>
      <p>Visualizes what actions the model is choosing and how confident it is.</p>
      <ul>
        <li><strong>4 Decision Types</strong> — Steering, throttle/brake, camera, and special actions. Each has its own distribution chart.</li>
        <li><strong>Healthy distributions</strong> — Should roughly match the distribution from your gameplay data. If one action dominates (e.g., always full throttle), the model may be mode-collapsing.</li>
        <li><strong>Entropy</strong> — Higher entropy means more exploration. Early training should have high entropy; it should decrease as the model becomes more confident.</li>
      </ul>
    </section>

    <!-- Hyperparams View -->
    <section :id="'hyperparams'" class="help-section">
      <h3>Hyperparams View</h3>
      <p>Live hyperparameter editor with hot-reload. Changes take effect on the next training step without restarting.</p>
      <ul>
        <li><strong>Learning Rate</strong> — Controls step size. Too high causes instability (loss spikes), too low causes slow convergence. Safe range: 1e-5 to 1e-3.</li>
        <li><strong>Batch Size</strong> — Larger batches give smoother gradients but use more memory. Typical: 16-128.</li>
        <li><strong>Discount (gamma)</strong> — How much the model values future rewards. 0.95-0.99 is typical for GTA tasks.</li>
        <li><strong>Entropy Coefficient</strong> — Encourages exploration. Start high (0.01-0.05), decrease as training matures.</li>
      </ul>
      <p>Edit a value and click Apply. The training loop picks up changes within one step.</p>
    </section>

    <!-- Sessions View -->
    <section :id="'sessions'" class="help-section">
      <h3>Sessions View</h3>
      <p>Compare metrics across multiple training sessions to evaluate the impact of changes.</p>
      <ul>
        <li><strong>Session list</strong> — Each session records start time, module, and final metrics.</li>
        <li><strong>Overlay mode</strong> — Select two sessions to overlay their loss/reward curves on the same chart. Useful for A/B comparisons after hyperparameter changes.</li>
        <li><strong>What to look for</strong> — A good change shows faster convergence (steeper initial slope) or higher final performance (asymptote) compared to baseline.</li>
      </ul>
    </section>

    <!-- Embeddings View -->
    <section :id="'embeddings'" class="help-section">
      <h3>Embeddings View</h3>
      <p>PCA projection of the model's internal representations, reduced to 2D for visualization.</p>
      <ul>
        <li><strong>Step coloring</strong> — Points colored by training step show how representations evolve. Clusters that form and tighten over time indicate the model is learning structure.</li>
        <li><strong>Context coloring</strong> — Points colored by game context (driving, intersection, combat) reveal whether the model distinguishes situations internally.</li>
        <li><strong>What to look for</strong> — Clear separation between contexts is good. If everything is one blob, the model isn't differentiating situations yet.</li>
      </ul>
    </section>

    <!-- Predictions View -->
    <section :id="'predictions'" class="help-section">
      <h3>Predictions View</h3>
      <p>Shows how well the model predicts outcomes versus what actually happened.</p>
      <ul>
        <li><strong>Scatter plot</strong> — Predicted vs actual values. Points on the diagonal mean perfect predictions. Spread indicates uncertainty.</li>
        <li><strong>MSE over time</strong> — Mean squared error trend. Should decrease as training progresses.</li>
        <li><strong>What to look for</strong> — Systematic bias (all points above/below diagonal) suggests the model has a consistent error pattern to fix.</li>
      </ul>
    </section>

    <!-- Weights View -->
    <section :id="'weights'" class="help-section">
      <h3>Weights View</h3>
      <p>Manage model checkpoints — snapshots of the model at specific training steps.</p>
      <ul>
        <li><strong>Checkpoint list</strong> — Shows step number, timestamp, and key metrics at save time.</li>
        <li><strong>Download</strong> — Export a checkpoint for deployment or backup.</li>
        <li><strong>Restore</strong> — Roll back training to a previous checkpoint. Useful if you notice degradation after a change.</li>
      </ul>
      <p>Checkpoints are saved automatically at configurable intervals and whenever you manually trigger a save.</p>
    </section>

    <!-- Glossary -->
    <section :id="'glossary'" class="help-section">
      <h3>Glossary</h3>
      <dl class="glossary">
        <dt>Token</dt><dd>A discrete unit of input (a quantized frame patch or action code) that the model processes.</dd>
        <dt>Embedding</dt><dd>A dense vector representation learned by the model to encode meaning from tokens.</dd>
        <dt>MCTS</dt><dd>Monte Carlo Tree Search — a planning algorithm that explores possible action sequences to find optimal decisions.</dd>
        <dt>Drift</dt><dd>When the model's behavior slowly shifts away from the training distribution, often due to distribution shift in new data.</dd>
        <dt>Convergence</dt><dd>When the loss stabilizes at a minimum and further training yields diminishing returns.</dd>
        <dt>Episode</dt><dd>One complete sequence of gameplay from start to terminal state (crash, mission end, etc.).</dd>
        <dt>Entropy</dt><dd>A measure of randomness in the model's action distribution. High entropy = more exploration.</dd>
        <dt>Checkpoint</dt><dd>A saved snapshot of model weights at a specific training step.</dd>
        <dt>PCA</dt><dd>Principal Component Analysis — a dimensionality reduction technique used to visualize high-dimensional embeddings in 2D.</dd>
        <dt>Gamma (discount)</dt><dd>How much future rewards are weighted relative to immediate rewards. Higher gamma = more forward-looking.</dd>
        <dt>Batch</dt><dd>A group of training samples processed together in one gradient update step.</dd>
        <dt>Hot-reload</dt><dd>Applying configuration changes to a running process without restarting it.</dd>
      </dl>
    </section>
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.view-header { display: flex; align-items: center; justify-content: space-between; }

.help-toc {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs);
  padding: var(--space-sm);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md, 6px);
}
.help-toc a {
  padding: var(--space-xs) var(--space-sm);
  color: var(--color-text-secondary);
  text-decoration: none;
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  font-size: var(--font-size-label);
}
.help-toc a:hover { color: var(--color-text-primary); background: var(--color-bg-active); }
.help-toc a.active { color: var(--color-accent); background: var(--color-bg-active); }

.help-section {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md, 6px);
  padding: var(--space-lg);
}
.help-section h3 {
  color: var(--color-text-primary);
  margin-bottom: var(--space-sm);
  font-size: var(--font-size-heading, 1.25rem);
}
.help-section p, .help-section li, .help-section dd {
  color: var(--color-text-secondary);
  line-height: 1.6;
}
.help-section ol, .help-section ul {
  padding-left: var(--space-lg);
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
}
.help-section code {
  background: var(--color-bg-active);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.9em;
}
.help-section strong { color: var(--color-text-primary); }

.glossary { display: grid; gap: var(--space-sm); }
.glossary dt {
  color: var(--color-accent);
  font-weight: var(--font-weight-semibold);
  margin-top: var(--space-sm);
}
.glossary dd {
  margin-left: var(--space-md);
  color: var(--color-text-secondary);
}
</style>
