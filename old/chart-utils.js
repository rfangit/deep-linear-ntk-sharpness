// ============================================================================
// CHART UTILITIES - Shared formatting and options for Chart.js plots
// ============================================================================

export const CHART_FONT = { family: 'Monaco, Consolas, "Courier New", monospace' };

export function formatTickLabel(value) {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs >= 1 && Math.abs(value - Math.round(value)) < 0.01) {
    return String(Math.round(value));
  }
  return parseFloat(value.toPrecision(4)).toString();
}

export function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        type: 'linear',
        min: 0,
        ticks: {
          maxRotation: 0,
          font: { size: 14, ...CHART_FONT },
          callback: function(value) { return formatTickLabel(value); }
        }
      },
      y: {
        type: 'linear',
        beginAtZero: true,
        ticks: {
          font: { size: 14, ...CHART_FONT },
          callback: function(value) { return formatTickLabel(value); }
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        align: 'start',
        onClick: () => {},
        labels: {
          usePointStyle: false,
          boxWidth: 40,
          boxHeight: 2,
          font: { size: 12, ...CHART_FONT }
        }
      }
    }
  };
}
