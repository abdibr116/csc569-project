<template>
  <div class="cards">
    <div v-for="method in comparison" :key="method.Method"
         :class="['card', cardClass(method.Method)]">
      <div class="card-label">{{ method.Method }}</div>
      <div class="card-f1">{{ method['F1 (weighted)'].toFixed(4) }}</div>
      <div class="card-sublabel">F1 (weighted)</div>
      <div class="card-details">
        <span>Acc: {{ method.Accuracy.toFixed(4) }}</span>
        <span>Time: {{ formatTime(method['Time (s)']) }}</span>
      </div>
      <div v-if="method['% F1 Improvement vs Default'] !== 0" class="card-delta"
           :class="method['% F1 Improvement vs Default'] > 0 ? 'positive' : 'negative'">
        {{ method['% F1 Improvement vs Default'] > 0 ? '+' : '' }}{{ method['% F1 Improvement vs Default'].toFixed(2) }}% vs Default
      </div>
    </div>

    <div class="card card-config">
      <div class="card-label">Experiment</div>
      <div class="config-items">
        <div><strong>Dataset:</strong> {{ config.dataset_source || 'unknown' }}</div>
        <div><strong>Samples:</strong> {{ config.n_samples?.toLocaleString() }}</div>
        <div><strong>CV:</strong> {{ config.ga_config?.pop_size ? `${statistical.n_folds || 10}-fold` : '' }}</div>
        <div><strong>Seeds:</strong> {{ config.n_seeds || 1 }}</div>
        <div>
          <strong>Wilcoxon:</strong>
          p={{ statistical.p_value?.toFixed(4) }}
          <span :class="statistical.significant_at_005 ? 'sig' : 'not-sig'">
            {{ statistical.significant_at_005 ? 'significant' : 'not significant' }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: ['comparison', 'config', 'statistical'],
  methods: {
    cardClass(method) {
      if (method === 'Default RF') return 'default'
      if (method === 'RandomizedSearchCV') return 'rs'
      return 'gp'
    },
    formatTime(s) {
      if (s < 60) return `${s.toFixed(1)}s`
      if (s < 3600) return `${(s / 60).toFixed(1)}m`
      return `${(s / 3600).toFixed(1)}h`
    }
  }
}
</script>

<style scoped>
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 30px; }
.card { background: #fff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid #ccc; }
.card.default { border-top-color: #2196F3; }
.card.rs { border-top-color: #FF9800; }
.card.gp { border-top-color: #4CAF50; }
.card.card-config { border-top-color: #9C27B0; }
.card-label { font-size: 0.85em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
.card-f1 { font-size: 2em; font-weight: 700; margin: 6px 0 2px; }
.card-sublabel { font-size: 0.8em; color: #aaa; }
.card-details { margin-top: 12px; font-size: 0.85em; color: #666; display: flex; justify-content: space-between; }
.card-delta { margin-top: 8px; font-size: 0.85em; font-weight: 600; }
.card-delta.positive { color: #4CAF50; }
.card-delta.negative { color: #f44336; }
.config-items { margin-top: 10px; font-size: 0.88em; line-height: 1.8; }
.sig { color: #4CAF50; font-weight: 600; }
.not-sig { color: #FF9800; font-weight: 600; }
</style>
