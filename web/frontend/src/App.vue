<template>
  <div class="app">
    <header>
      <h1>RF Hyperparameter Optimization via Genetic Algorithm</h1>
      <p class="subtitle">CSC 569 - Results Dashboard</p>
    </header>

    <ExperimentPanel
      :experiment="experiment"
      @start="startExperiment"
      @stop="stopExperiment"
      @refresh="refreshResults"
      @dismiss="dismissExperiment"
    />

    <div v-if="loading" class="loading">Loading results...</div>

    <template v-else>
      <SummaryCards :comparison="comparison" :config="config" :statistical="statistical" />

      <div class="tabs">
        <button v-for="tab in tabs" :key="tab.id"
                :class="{ active: activeTab === tab.id }"
                @click="activeTab = tab.id">
          {{ tab.label }}
        </button>
      </div>

      <div class="tab-content">
        <ComparisonTab v-if="activeTab === 'comparison'" :comparison="comparison" />
        <EvolutionTab v-if="activeTab === 'evolution'" :evolution="evolution" :gp="gp" />
        <ConfusionTab v-if="activeTab === 'confusion'" :confusion="confusion" />
        <FeaturesTab v-if="activeTab === 'features'" :features="features" />
        <SeedsTab v-if="activeTab === 'seeds'" :seeds="seeds" :statistical="statistical" />
        <PaperTab v-if="activeTab === 'paper'" :sections="paperSections" />
      </div>
    </template>
  </div>
</template>

<script>
import SummaryCards from './components/SummaryCards.vue'
import ComparisonTab from './components/ComparisonTab.vue'
import EvolutionTab from './components/EvolutionTab.vue'
import ConfusionTab from './components/ConfusionTab.vue'
import FeaturesTab from './components/FeaturesTab.vue'
import SeedsTab from './components/SeedsTab.vue'
import PaperTab from './components/PaperTab.vue'
import ExperimentPanel from './components/ExperimentPanel.vue'

const API = '/api'

export default {
  components: { SummaryCards, ComparisonTab, EvolutionTab, ConfusionTab, FeaturesTab, SeedsTab, PaperTab, ExperimentPanel },
  data() {
    return {
      loading: true,
      activeTab: 'comparison',
      tabs: [
        { id: 'comparison', label: 'Comparison' },
        { id: 'evolution', label: 'Evolution' },
        { id: 'confusion', label: 'Confusion Matrix' },
        { id: 'features', label: 'Features' },
        { id: 'seeds', label: 'Seeds' },
        { id: 'paper', label: 'Paper' },
      ],
      comparison: [],
      config: {},
      gp: {},
      evolution: [],
      confusion: { matrix: [], labels: [] },
      features: {},
      statistical: {},
      seeds: {},
      paperSections: [],
      experiment: { status: 'idle' },
      experimentPoller: null,
    }
  },
  async mounted() {
    await this.loadResults()
    await this.checkExperiment()
  },
  beforeUnmount() {
    if (this.experimentPoller) clearInterval(this.experimentPoller)
  },
  methods: {
    async loadResults() {
      try {
        const endpoints = ['comparison', 'config', 'gp', 'evolution', 'confusion', 'features', 'statistical', 'seeds', 'paper']
        const results = await Promise.allSettled(
          endpoints.map(e => fetch(`${API}/${e}`).then(r => r.ok ? r.json() : null))
        )
        const [comp, cfg, gp, evo, conf, feat, stat, seedData, paper] = results.map(r => r.status === 'fulfilled' ? r.value : null)
        if (comp) this.comparison = comp
        if (cfg) this.config = cfg
        if (gp) this.gp = gp
        if (evo) this.evolution = evo
        if (conf) this.confusion = conf
        if (feat) this.features = feat
        if (stat) this.statistical = stat
        if (seedData) this.seeds = seedData
        if (paper) this.paperSections = paper
      } catch (e) {
        console.error('Failed to load results:', e)
      } finally {
        this.loading = false
      }
    },

    async checkExperiment() {
      try {
        const res = await fetch(`${API}/experiment/status`)
        const prev = this.experiment.status
        this.experiment = await res.json()

        if (this.experiment.status === 'running' && !this.experimentPoller) {
          this.experimentPoller = setInterval(() => this.checkExperiment(), 3000)
        }
        if (this.experiment.status !== 'running' && this.experimentPoller) {
          clearInterval(this.experimentPoller)
          this.experimentPoller = null
        }
        if (prev === 'running' && this.experiment.status === 'completed') {
          await this.loadResults()
        }
      } catch (e) {
        console.error('Failed to check experiment:', e)
      }
    },

    async startExperiment(config) {
      try {
        const res = await fetch(`${API}/experiment/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        })
        if (!res.ok) {
          const err = await res.json()
          throw new Error(err.detail || 'Failed to start')
        }
        await this.checkExperiment()
      } catch (e) {
        console.error('Start failed:', e)
        alert('Failed to start experiment: ' + e.message)
      }
    },

    async stopExperiment() {
      if (!confirm('Stop the running experiment?')) return
      try {
        await fetch(`${API}/experiment/stop`, { method: 'POST' })
        this.experiment = { status: 'idle' }
      } catch (e) {
        console.error('Stop failed:', e)
      }
    },

    async refreshResults() {
      this.loading = true
      await this.loadResults()
      this.experiment = { status: 'idle' }
    },

    async dismissExperiment() {
      try {
        await fetch(`${API}/experiment`, { method: 'DELETE' })
      } catch (e) { /* ignore */ }
      this.experiment = { status: 'idle' }
    },
  },
}
</script>

<style>
.app { max-width: 1200px; margin: 0 auto; padding: 20px; }
header { text-align: center; margin-bottom: 30px; }
header h1 { font-size: 1.6em; color: #1a237e; }
.subtitle { color: #666; margin-top: 4px; }
.loading { text-align: center; padding: 60px; color: #888; font-size: 1.2em; }
.tabs { display: flex; gap: 4px; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px; flex-wrap: wrap; }
.tabs button {
  padding: 10px 20px; border: none; background: none; cursor: pointer;
  font-size: 0.95em; color: #666; border-bottom: 2px solid transparent;
  margin-bottom: -2px; transition: all 0.2s;
}
.tabs button.active { color: #1a237e; border-bottom-color: #1a237e; font-weight: 600; }
.tabs button:hover { color: #333; }
.tab-content { min-height: 400px; }
</style>
