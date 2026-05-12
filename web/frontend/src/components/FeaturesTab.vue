<template>
  <div>
    <h2>Feature Importance - GP-Optimized Model</h2>
    <div ref="barChart" class="chart"></div>
  </div>
</template>

<script>
import Plotly from 'plotly.js-dist-min'

export default {
  props: ['features'],
  methods: {
    drawChart() {
      if (!Object.keys(this.features).length) return
      const sorted = Object.entries(this.features).sort((a, b) => a[1] - b[1])
      Plotly.newPlot(this.$refs.barChart, [{
        y: sorted.map(([k]) => k),
        x: sorted.map(([, v]) => v),
        type: 'bar', orientation: 'h',
        marker: { color: '#4CAF50' },
        text: sorted.map(([, v]) => v.toFixed(4)),
        textposition: 'outside',
      }], {
        title: 'Feature Importance - GP-Optimized Random Forest',
        xaxis: { title: 'Importance' },
        margin: { l: 140, t: 50, r: 60 },
        height: 500,
      }, { responsive: true })
    }
  },
  mounted() { this.drawChart() },
  watch: { features() { this.drawChart() } }
}
</script>

<style scoped>
h2 { color: #1a237e; margin-bottom: 16px; }
.chart { width: 100%; height: 520px; }
</style>
