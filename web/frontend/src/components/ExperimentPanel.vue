<template>
  <div class="experiment-panel">
    <!-- Idle: show run button -->
    <div v-if="status === 'idle' && !showForm" class="idle-bar">
      <button class="run-btn" @click="showForm = true">Configure & Run Experiment</button>
    </div>

    <!-- Config form -->
    <div v-if="showForm && status === 'idle'" class="config-form">
      <h3>Configure Experiment</h3>
      <div class="form-grid">
        <label>
          <span>Population Size</span>
          <input type="number" v-model.number="form.pop_size" min="2" max="200" />
        </label>
        <label>
          <span>Generations</span>
          <input type="number" v-model.number="form.generations" min="1" max="100" />
        </label>
        <label class="full-width">
          <span>Seeds (comma-separated)</span>
          <input type="text" v-model="form.seedsText" placeholder="42, 123, 456, 789, 1024" />
        </label>
      </div>
      <div class="form-actions">
        <button class="start-btn" @click="startExperiment" :disabled="starting">
          {{ starting ? 'Starting...' : 'Start Experiment' }}
        </button>
        <button class="cancel-btn" @click="showForm = false">Cancel</button>
      </div>
      <p v-if="error" class="error">{{ error }}</p>
    </div>

    <!-- Running: progress display -->
    <div v-if="status === 'running'" class="running-panel">
      <div class="running-header">
        <h3>Experiment Running</h3>
        <button class="stop-btn" @click="$emit('stop')">Stop</button>
      </div>

      <div class="step-pills">
        <span v-for="step in allSteps" :key="step"
              :class="['pill', stepClass(step)]">
          {{ stepLabel(step) }}
        </span>
      </div>

      <div v-if="experiment.gp_progress" class="gp-section">
        <div class="progress-bar-wrap">
          <div class="progress-bar" :style="{ width: progressPct + '%' }"></div>
          <span class="progress-label">{{ progressPct.toFixed(0) }}%</span>
        </div>
        <div class="gp-stats">
          <span>Seed {{ experiment.gp_progress.current_seed }}</span>
          <span>Gen {{ experiment.gp_progress.current_gen }}/{{ experiment.gp_progress.total_gen }}</span>
          <span v-if="experiment.gp_progress.fitness_max">Best F1: {{ experiment.gp_progress.fitness_max.toFixed(4) }}</span>
          <span v-if="experiment.gp_progress.fitness_avg">Avg F1: {{ experiment.gp_progress.fitness_avg.toFixed(4) }}</span>
        </div>
      </div>

      <div class="eta-row" v-if="experiment.eta">
        <span v-if="experiment.eta.estimated_remaining_human">
          ETA: <strong>{{ experiment.eta.estimated_remaining_human }}</strong> remaining
        </span>
        <span v-else>Calculating ETA...</span>
        <span class="elapsed" v-if="experiment.eta.completed_gens !== undefined">
          ({{ experiment.eta.completed_gens }}/{{ experiment.eta.total_gens }} generations)
        </span>
      </div>
    </div>

    <!-- Completed -->
    <div v-if="status === 'completed'" class="completed-panel">
      <div class="completed-header">
        <span class="check-icon">&#10003;</span>
        <span>Experiment Complete!</span>
        <span v-if="experiment.total_time_human" class="time">Finished in {{ experiment.total_time_human }}</span>
      </div>
      <div class="completed-actions">
        <button class="refresh-btn" @click="$emit('refresh')">Refresh Results</button>
        <button class="dismiss-btn" @click="$emit('dismiss')">Dismiss</button>
      </div>
    </div>

    <!-- Failed -->
    <div v-if="status === 'failed'" class="failed-panel">
      <div class="failed-header">
        <span>Experiment Failed</span>
      </div>
      <p class="error" v-if="experiment.error">{{ experiment.error }}</p>
      <button class="dismiss-btn" @click="$emit('dismiss')">Dismiss</button>
    </div>
  </div>
</template>

<script>
export default {
  props: ['experiment'],
  emits: ['start', 'stop', 'refresh', 'dismiss'],
  data() {
    return {
      showForm: false,
      starting: false,
      error: null,
      form: {
        pop_size: 50,
        generations: 30,
        seedsText: '42, 123, 456, 789, 1024',
      },
      allSteps: ['download', 'preprocess', 'baseline', 'gp', 'aggregate', 'evaluate', 'draft_paper'],
    }
  },
  computed: {
    status() {
      return this.experiment?.status || 'idle'
    },
    progressPct() {
      const eta = this.experiment?.eta
      if (!eta || !eta.total_gens) return 0
      return Math.min(100, (eta.completed_gens / eta.total_gens) * 100)
    },
  },
  methods: {
    stepLabel(step) {
      const labels = {
        download: 'Download', preprocess: 'Preprocess', baseline: 'Baseline',
        gp: 'GP', aggregate: 'Aggregate', evaluate: 'Evaluate', draft_paper: 'Paper',
      }
      return labels[step] || step
    },
    stepClass(step) {
      const completed = this.experiment?.steps_completed || []
      const current = this.experiment?.current_step
      if (completed.some(s => s === step || s.startsWith(step + '_seed_'))) return 'done'
      if (current === step) return 'active'
      return 'pending'
    },
    async startExperiment() {
      this.error = null
      const seeds = this.form.seedsText.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
      if (!seeds.length) {
        this.error = 'Enter at least one valid seed number'
        return
      }
      this.starting = true
      try {
        this.$emit('start', {
          pop_size: this.form.pop_size,
          generations: this.form.generations,
          seeds: seeds,
        })
        this.showForm = false
      } catch (e) {
        this.error = e.message
      } finally {
        this.starting = false
      }
    },
  },
}
</script>

<style scoped>
.experiment-panel { margin-bottom: 24px; }

.idle-bar { text-align: center; }
.run-btn {
  padding: 10px 28px; background: #1a237e; color: #fff; border: none;
  border-radius: 8px; font-size: 0.95em; cursor: pointer; transition: background 0.2s;
}
.run-btn:hover { background: #283593; }

.config-form {
  background: #fff; border-radius: 10px; padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #1a237e;
}
.config-form h3 { margin: 0 0 16px; color: #1a237e; }
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.form-grid label { display: flex; flex-direction: column; gap: 4px; }
.form-grid label span { font-size: 0.85em; color: #666; }
.form-grid input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95em; }
.full-width { grid-column: 1 / -1; }
.form-actions { margin-top: 16px; display: flex; gap: 10px; }
.start-btn {
  padding: 10px 24px; background: #4CAF50; color: #fff; border: none;
  border-radius: 6px; font-size: 0.95em; cursor: pointer;
}
.start-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.cancel-btn { padding: 10px 24px; background: #eee; border: none; border-radius: 6px; cursor: pointer; }
.error { color: #f44336; margin-top: 8px; font-size: 0.9em; }

.running-panel {
  background: #fff; border-radius: 10px; padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #2196F3;
}
.running-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.running-header h3 { margin: 0; color: #1a237e; }
.stop-btn {
  padding: 6px 16px; background: #f44336; color: #fff; border: none;
  border-radius: 6px; font-size: 0.85em; cursor: pointer;
}

.step-pills { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
.pill {
  padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: 500;
}
.pill.pending { background: #f5f5f5; color: #aaa; }
.pill.active { background: #e3f2fd; color: #1565c0; animation: pulse 1.5s infinite; }
.pill.done { background: #e8f5e9; color: #2e7d32; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

.progress-bar-wrap {
  height: 24px; background: #e0e0e0; border-radius: 12px;
  position: relative; overflow: hidden; margin-bottom: 10px;
}
.progress-bar {
  height: 100%; background: linear-gradient(90deg, #4CAF50, #66BB6A);
  border-radius: 12px; transition: width 0.5s ease;
}
.progress-label {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  font-size: 0.8em; font-weight: 600; color: #333;
}
.gp-stats { display: flex; gap: 20px; font-size: 0.88em; color: #555; flex-wrap: wrap; }
.eta-row { margin-top: 12px; font-size: 0.9em; color: #666; }
.elapsed { margin-left: 8px; color: #aaa; }

.completed-panel {
  background: #e8f5e9; border-radius: 10px; padding: 20px;
  border-left: 4px solid #4CAF50;
}
.completed-header { display: flex; align-items: center; gap: 10px; font-size: 1.05em; }
.check-icon { color: #4CAF50; font-size: 1.3em; font-weight: bold; }
.time { margin-left: auto; color: #666; font-size: 0.9em; }
.completed-actions { margin-top: 12px; display: flex; gap: 10px; }
.refresh-btn {
  padding: 8px 20px; background: #4CAF50; color: #fff; border: none;
  border-radius: 6px; cursor: pointer;
}
.dismiss-btn { padding: 8px 20px; background: #eee; border: none; border-radius: 6px; cursor: pointer; }

.failed-panel {
  background: #fce4ec; border-radius: 10px; padding: 20px;
  border-left: 4px solid #f44336;
}
.failed-header { font-size: 1.05em; color: #c62828; font-weight: 600; margin-bottom: 8px; }
</style>
