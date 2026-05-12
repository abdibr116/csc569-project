<template>
  <div>
    <h2>Multi-Seed Analysis</h2>

    <div v-if="!seeds.summary || !seeds.summary.length" class="empty">
      Single-seed run detected. Run with <code>--seeds 42 123 456 789 1024</code> to see multi-seed analysis.
    </div>

    <template v-else>
      <div class="seed-summary">
        <div class="stat-card">
          <div class="stat-label">Seeds</div>
          <div class="stat-value">{{ seeds.n_seeds }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Best Seed</div>
          <div class="stat-value">{{ seeds.best_seed }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">F1 Across Seeds</div>
          <div class="stat-value">{{ seeds.f1_mean_across_seeds?.toFixed(4) }} <span class="pm">+/- {{ seeds.f1_std_across_seeds?.toFixed(4) }}</span></div>
        </div>
      </div>

      <h3>Per-Seed Results</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Seed</th><th>Best Fitness</th><th>CV F1 Mean</th><th>CV F1 Std</th><th>Time (s)</th></tr>
          </thead>
          <tbody>
            <tr v-for="s in seeds.summary" :key="s.seed" :class="{ best: s.seed === seeds.best_seed }">
              <td>{{ s.seed }} {{ s.seed === seeds.best_seed ? ' (best)' : '' }}</td>
              <td>{{ s.best_fitness.toFixed(4) }}</td>
              <td>{{ s.cv_f1_mean.toFixed(4) }}</td>
              <td>{{ s.cv_f1_std.toFixed(4) }}</td>
              <td>{{ s.time_seconds.toFixed(1) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>Statistical Tests Per Seed</h3>
      <div v-if="statistical.per_seed_tests" class="table-wrap">
        <table>
          <thead>
            <tr><th>Seed</th><th>Statistic</th><th>p-value</th><th>Significant (p&lt;0.05)</th></tr>
          </thead>
          <tbody>
            <tr v-for="t in statistical.per_seed_tests" :key="t.seed">
              <td>{{ t.seed }}</td>
              <td>{{ t.statistic?.toFixed(2) ?? 'N/A' }}</td>
              <td>{{ t.p_value.toFixed(4) }}</td>
              <td :class="t.significant_at_0_05 ? 'sig' : 'not-sig'">
                {{ t.significant_at_0_05 ? 'Yes' : 'No' }}
              </td>
            </tr>
          </tbody>
        </table>
        <p class="summary-line">
          {{ statistical.seeds_significant_count }}/{{ statistical.per_seed_tests.length }} seeds reached significance at alpha = 0.05
        </p>
      </div>
    </template>
  </div>
</template>

<script>
export default {
  props: ['seeds', 'statistical'],
}
</script>

<style scoped>
h2, h3 { color: #1a237e; }
h3 { margin: 24px 0 12px; }
.empty { background: #fff3e0; padding: 20px; border-radius: 8px; color: #e65100; }
code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
.seed-summary { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
.stat-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 24px; }
.stat-label { font-size: 0.8em; color: #888; text-transform: uppercase; }
.stat-value { font-size: 1.5em; font-weight: 700; color: #1a237e; }
.pm { font-size: 0.6em; color: #888; font-weight: 400; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f5f5f5; font-weight: 600; }
tr.best { background: #e8f5e9; }
.sig { color: #4CAF50; font-weight: 600; }
.not-sig { color: #FF9800; font-weight: 600; }
.summary-line { margin-top: 12px; color: #666; font-style: italic; }
</style>
