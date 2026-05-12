<template>
  <div>
    <h2>Performance Comparison</h2>
    <div ref="barChart" class="chart"></div>
    <h3>Full Metrics Table</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 (weighted)</th>
            <th>CV F1 Mean +/- Std</th>
            <th>Time</th>
            <th>F1 vs Default</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in comparison" :key="row.Method">
            <td><strong>{{ row.Method }}</strong></td>
            <td>{{ row.Accuracy.toFixed(4) }}</td>
            <td>{{ row.Precision.toFixed(4) }}</td>
            <td>{{ row.Recall.toFixed(4) }}</td>
            <td>{{ row['F1 (weighted)'].toFixed(4) }}</td>
            <td>{{ row['CV F1 Mean +/- Std'] }}</td>
            <td>{{ formatTime(row['Time (s)']) }}</td>
            <td :class="deltaClass(row['% F1 Improvement vs Default'])">
              {{ row['% F1 Improvement vs Default'] === 0 ? '--' : (row['% F1 Improvement vs Default'] > 0 ? '+' : '') + row['% F1 Improvement vs Default'].toFixed(2) + '%' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import Plotly from 'plotly.js-dist-min'

export default {
  props: ['comparison'],
  methods: {
    formatTime(s) {
      if (s < 60) return `${s.toFixed(1)}s`
      if (s < 3600) return `${(s / 60).toFixed(1)}m`
      return `${(s / 3600).toFixed(1)}h`
    },
    deltaClass(val) {
      if (val > 0) return 'positive'
      if (val < 0) return 'negative'
      return ''
    },
    drawChart() {
      const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (weighted)']
      const colors = { 'Default RF': '#2196F3', 'RandomizedSearchCV': '#FF9800', 'GP-Optimized': '#4CAF50' }
      const traces = this.comparison.map(row => ({
        x: metrics,
        y: metrics.map(m => row[m]),
        name: row.Method,
        type: 'bar',
        marker: { color: colors[row.Method] || '#888' },
        text: metrics.map(m => row[m].toFixed(4)),
        textposition: 'outside',
        textfont: { size: 11 },
      }))
      const allVals = this.comparison.flatMap(row => metrics.map(m => row[m]))
      Plotly.newPlot(this.$refs.barChart, traces, {
        barmode: 'group',
        title: 'Performance Comparison Across Tuning Methods',
        yaxis: { range: [Math.max(0, Math.min(...allVals) - 0.03), 1.0], title: 'Score' },
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 50, b: 80 },
      }, { responsive: true })
    }
  },
  mounted() { this.drawChart() },
  watch: { comparison() { this.drawChart() } }
}
</script>

<style scoped>
h2 { margin-bottom: 16px; color: #1a237e; }
h3 { margin: 24px 0 12px; color: #333; }
.chart { width: 100%; height: 450px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f5f5f5; font-weight: 600; white-space: nowrap; }
.positive { color: #4CAF50; font-weight: 600; }
.negative { color: #f44336; font-weight: 600; }
</style>
