<template>
  <div class="app">
    <header>
      <h1>RF Hyperparameter Optimization via Genetic Algorithm</h1>
      <p class="subtitle">CSC 569 - Results Dashboard</p>
    </header>

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

const API = '/api'

export default {
  components: { SummaryCards, ComparisonTab, EvolutionTab, ConfusionTab, FeaturesTab, SeedsTab, PaperTab },
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
    }
  },
  async mounted() {
    try {
      const [comp, cfg, gp, evo, conf, feat, stat, seedData, paper] = await Promise.all([
        fetch(`${API}/comparison`).then(r => r.json()),
        fetch(`${API}/config`).then(r => r.json()),
        fetch(`${API}/gp`).then(r => r.json()),
        fetch(`${API}/evolution`).then(r => r.json()),
        fetch(`${API}/confusion`).then(r => r.json()),
        fetch(`${API}/features`).then(r => r.json()),
        fetch(`${API}/statistical`).then(r => r.json()),
        fetch(`${API}/seeds`).then(r => r.json()),
        fetch(`${API}/paper`).then(r => r.json()),
      ])
      this.comparison = comp
      this.config = cfg
      this.gp = gp
      this.evolution = evo
      this.confusion = conf
      this.features = feat
      this.statistical = stat
      this.seeds = seedData
      this.paperSections = paper
    } catch (e) {
      console.error('Failed to load data:', e)
    } finally {
      this.loading = false
    }
  }
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
