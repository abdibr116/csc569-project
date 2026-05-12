<template>
  <div>
    <h2>Paper Sections (Auto-Generated)</h2>
    <div class="section-nav">
      <button v-for="s in sections" :key="s.section"
              :class="{ active: activeSection === s.section }"
              @click="activeSection = s.section">
        {{ s.section }}. {{ sectionTitle(s.section) }}
      </button>
    </div>
    <div class="paper-content" v-for="s in sections" :key="s.section" v-show="activeSection === s.section">
      <div v-html="s.html"></div>
    </div>
  </div>
</template>

<script>
export default {
  props: ['sections'],
  data() {
    return { activeSection: '06' }
  },
  methods: {
    sectionTitle(num) {
      const titles = { '06': 'Experiments', '07': 'Results', '08': 'Discussion', '09': 'Challenges', '10': 'Future Work' }
      return titles[num] || ''
    }
  }
}
</script>

<style scoped>
h2 { color: #1a237e; margin-bottom: 16px; }
.section-nav { display: flex; gap: 6px; margin-bottom: 20px; flex-wrap: wrap; }
.section-nav button {
  padding: 8px 16px; border: 1px solid #e0e0e0; border-radius: 20px;
  background: #fff; cursor: pointer; font-size: 0.85em; transition: all 0.2s;
}
.section-nav button.active { background: #1a237e; color: #fff; border-color: #1a237e; }
.section-nav button:hover { border-color: #1a237e; }
.paper-content {
  background: #fff; border-radius: 10px; padding: 30px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  line-height: 1.7; font-size: 0.95em;
}
.paper-content :deep(h1) { font-size: 1.4em; color: #1a237e; margin-bottom: 16px; }
.paper-content :deep(h2) { font-size: 1.15em; color: #333; margin: 20px 0 10px; }
.paper-content :deep(table) { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
.paper-content :deep(th), .paper-content :deep(td) { padding: 8px 12px; border: 1px solid #e0e0e0; text-align: left; }
.paper-content :deep(th) { background: #f5f5f5; }
.paper-content :deep(code) { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
.paper-content :deep(pre) { background: #f5f5f5; padding: 16px; border-radius: 6px; overflow-x: auto; }
.paper-content :deep(strong) { color: #1a237e; }
</style>
