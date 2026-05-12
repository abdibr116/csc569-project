<template>
  <div>
    <h2>GP Fitness Evolution</h2>
    <div ref="fitnessChart" class="chart"></div>

    <h3>Best Configuration Found</h3>
    <div class="params-grid" v-if="gp.best_params">
      <div class="param" v-for="(val, key) in gp.best_params" :key="key">
        <span class="param-key">{{ key }}</span>
        <span class="param-val">{{ val === null ? 'None' : val }}</span>
      </div>
    </div>

    <h3>Evolution Log</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Gen</th><th>Evals</th><th>Avg F1</th><th>Std</th><th>Min</th><th>Max</th></tr>
        </thead>
        <tbody>
          <tr v-for="row in evolution" :key="row.gen">
            <td>{{ row.gen }}</td>
            <td>{{ row.nevals }}</td>
            <td>{{ row.avg.toFixed(4) }}</td>
            <td>{{ row.std.toFixed(4) }}</td>
            <td>{{ row.min.toFixed(4) }}</td>
            <td>{{ row.max.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import Plotly from 'plotly.js-dist-min'

export default {
  props: ['evolution', 'gp'],
  methods: {
    drawChart() {
      if (!this.evolution.length) return
      const gens = this.evolution.map(r => r.gen)
      const best = { x: gens, y: this.evolution.map(r => r.max), name: 'Best Fitness', type: 'scatter', line: { color: '#4CAF50', width: 2.5 } }
      const avg = { x: gens, y: this.evolution.map(r => r.avg), name: 'Average Fitness', type: 'scatter', line: { color: '#2196F3', width: 2 } }
      const upper = this.evolution.map(r => r.avg + r.std)
      const lower = this.evolution.map(r => r.avg - r.std)
      const band = {
        x: [...gens, ...gens.slice().reverse()],
        y: [...upper, ...lower.reverse()],
        fill: 'toself', fillcolor: 'rgba(33,150,243,0.15)',
        line: { color: 'transparent' }, name: 'Avg +/- Std', showlegend: true, type: 'scatter',
      }
      Plotly.newPlot(this.$refs.fitnessChart, [band, avg, best], {
        title: 'GP Fitness Evolution Over Generations',
        xaxis: { title: 'Generation' },
        yaxis: { title: 'F1 Score (weighted)' },
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 50, b: 80 },
      }, { responsive: true })
    }
  },
  mounted() { this.drawChart() },
  watch: { evolution() { this.drawChart() } }
}
</script>

<style scoped>
h2, h3 { color: #1a237e; }
h3 { margin: 24px 0 12px; }
.chart { width: 100%; height: 450px; }
.params-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin: 12px 0; }
.param { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; }
.param-key { display: block; font-size: 0.8em; color: #888; text-transform: uppercase; }
.param-val { font-size: 1.3em; font-weight: 600; color: #1a237e; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f5f5f5; font-weight: 600; }
</style>
