// ============================================================================
// NTK DISPLAY - Render an NTK matrix as a light heatmap with values inside.
// ============================================================================
// API:
//
//   renderNtkMatrix(container, matrix, { rowLabels, colLabels })
//
// `container` is the DOM element to populate. `matrix` is a square 2D array
// (typically (N·n) × (N·n)). `rowLabels` / `colLabels` are arrays of LaTeX
// strings (e.g. "x = (1,0),\\ j = 1") that get rendered with MathJax.
//
// Color scheme: diverging red/white/blue heatmap, normalized by the matrix-
// wide max absolute value. Negative values shade red, positive blue, zero
// stays white. Numerical text on top stays readable thanks to muted endpoints.
//
// MathJax: we wrap labels in \(...\) and call typesetPromise on the freshly
// inserted nodes. The page is responsible for having loaded MathJax.

const CELL_PX = 86;       // square cell size in pixels
const FONT_PX = 15;
const HEADER_FONT = 13;
const BORDER_COLOR = '#ddd';
const HEADER_COLOR = '#666';

// Diverging ramp: red (negative) → white (zero) → blue (positive).
// `signedFraction` is value / maxAbs ∈ [-1, 1]. Light, muted endpoints so
// numerical text stays readable on top.
function colorFor(signedFraction) {
  const t = Math.max(-1, Math.min(1, signedFraction));
  // Negative end: muted red #e8a8a8.  Positive end: muted blue #a8c4e8.
  if (t >= 0) {
    const r = Math.round(255 + t * (168 - 255));   // 255 → 168
    const g = Math.round(255 + t * (196 - 255));   // 255 → 196
    const b = Math.round(255 + t * (232 - 255));   // 255 → 232
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const a = -t;                                  // 0..1
    const r = Math.round(255 + a * (232 - 255));   // 255 → 232
    const g = Math.round(255 + a * (168 - 255));   // 255 → 168
    const b = Math.round(255 + a * (168 - 255));   // 255 → 168
    return `rgb(${r}, ${g}, ${b})`;
  }
}

export function renderNtkMatrix(container, matrix, { rowLabels, colLabels }) {
  const n = matrix.length;

  // Find max |value| for normalization. Guard against the all-zero case.
  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = Math.abs(matrix[i][j]);
      if (v > maxAbs) maxAbs = v;
    }
  }
  if (maxAbs === 0) maxAbs = 1;

  // Build the table.
  const table = document.createElement('table');
  table.style.borderCollapse = 'separate';
  table.style.borderSpacing = '2px';
  table.style.fontFamily = "'Helvetica Neue', Helvetica, Arial, sans-serif";

  // Header row: empty corner + column labels.
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');

  const corner = document.createElement('th');
  corner.style.width = `${CELL_PX}px`;
  headerRow.appendChild(corner);

  for (let j = 0; j < n; j++) {
    const th = document.createElement('th');
    th.style.width = `${CELL_PX}px`;
    th.style.fontWeight = 'normal';
    th.style.fontSize = `${HEADER_FONT}px`;
    th.style.color = HEADER_COLOR;
    th.style.padding = '6px 4px';
    th.style.textAlign = 'center';
    th.innerHTML = `\\(${colLabels[j]}\\)`;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body: row label + n value cells.
  const tbody = document.createElement('tbody');
  for (let i = 0; i < n; i++) {
    const tr = document.createElement('tr');

    const rowTh = document.createElement('th');
    rowTh.style.fontWeight = 'normal';
    rowTh.style.fontSize = `${HEADER_FONT}px`;
    rowTh.style.color = HEADER_COLOR;
    rowTh.style.padding = '4px 8px';
    rowTh.style.textAlign = 'right';
    rowTh.style.whiteSpace = 'nowrap';
    rowTh.innerHTML = `\\(${rowLabels[i]}\\)`;
    tr.appendChild(rowTh);

    for (let j = 0; j < n; j++) {
      const td = document.createElement('td');
      const v = matrix[i][j];
      td.style.width = `${CELL_PX}px`;
      td.style.height = `${CELL_PX}px`;
      td.style.textAlign = 'center';
      td.style.verticalAlign = 'middle';
      td.style.fontSize = `${FONT_PX}px`;
      td.style.fontFamily = "'SF Mono', 'Menlo', 'Consolas', monospace";
      td.style.color = '#333';
      td.style.border = `1px solid ${BORDER_COLOR}`;
      td.style.borderRadius = '3px';
      td.style.backgroundColor = colorFor(v / maxAbs);
      td.textContent = formatNumber(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  container.innerHTML = '';
  container.appendChild(table);

  // Trigger MathJax on the new nodes.
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([container]).catch(() => {});
  }
}

function formatNumber(v) {
  // Two decimals, fixed-point. Render -0.00 as 0.00 to avoid the cosmetic
  // inconsistency.
  let s = v.toFixed(2);
  if (s === '-0.00') s = '0.00';
  return s;
}
