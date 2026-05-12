<template>
  <div>
    <h2>Confusion Matrix - GP-Optimized Model</h2>
    <div ref="heatmap" class="chart"></div>
  </div>
</template>

<script>
import Plotly from 'plotly.js-dist-min'

export default {
  props: ['confusion'],
  methods: {
    drawChart() {
      if (!this.confusion.matrix.length) return
      const labels = this.confusion.labels
      const z = this.confusion.matrix
      const annotations = []
      for (let i = 0; i < z.length; i++) {
        for (let j = 0; j < z[i].length; j++) {
          annotations.push({
            x: labels[j], y: labels[i], text: String(z[i][j]),
            showarrow: false, font: { color: z[i][j] > Math.max(...z.flat()) * 0.5 ? '#fff' : '#333', size: 13 }
          })
        }
      }
      Plotly.newPlot(this.$refs.heatmap, [{
        z: z, x: labels, y: labels, type: 'heatmap',
        colorscale: 'Blues', showscale: true,
      }], {
        title: 'Confusion Matrix - GP-Optimized Random Forest',
        xaxis: { title: 'Predicted', tickangle: -45 },
        yaxis: { title: 'True', autorange: 'reversed' },
        annotations: annotations,
        margin: { t: 50, b: 120, l: 120 },
        width: 700, height: 600,
      }, { responsive: true })
    }
  },
  mounted() { this.drawChart() },
  watch: { confusion: { handler() { this.drawChart() }, deep: true } }
}
</script>

<style scoped>
h2 { color: #1a237e; margin-bottom: 16px; }
.chart { width: 100%; max-width: 750px; height: 620px; margin: 0 auto; }
</style>
