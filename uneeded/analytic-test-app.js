// ============================================================================
// ANALYTIC-TEST-APP - Standalone page with just widgets 1 and 2.
// ============================================================================
// Sole purpose: side-by-side confirmation that the Jacobian-based NTK and the
// closed-form 2-layer NTK produce identical matrices.

import { randomModel } from './ntkModels.js';
import { ntkMatrix, ntkMatrixAnalytic2Layer } from './ntk.js';
import { renderNtkMatrix } from './ntkDisplay.js';

const INPUT_DIM = 2;
const HIDDEN_DIM = 2;
const OUTPUT_DIM = 2;
const DATA_POINTS = [[1, 0], [0, 1]];

function buildRowColLabels() {
  const labels = [];
  for (let a = 0; a < DATA_POINTS.length; a++) {
    const xText = `x{=}(${DATA_POINTS[a].join(',')})`;
    for (let j = 0; j < OUTPUT_DIM; j++) {
      labels.push(`${xText},\\ j{=}${j + 1}`);
    }
  }
  return labels;
}

function matrixToTeX(name, M) {
  let tex = `${name} = \\begin{bmatrix} `;
  tex += M.map(row => row.map(v => v.toFixed(2)).join(' & ')).join(' \\\\ ');
  tex += ' \\end{bmatrix}';
  return tex;
}

function makeWidget(prefix, ntkFn) {
  const refs = {
    seedInput:        document.getElementById(`${prefix}-model-seed`),
    generateBtn:      document.getElementById(`${prefix}-generate`),
    matrixContainer:  document.getElementById(`${prefix}-ntk-matrix`),
    W1Container:      document.getElementById(`${prefix}-W1`),
    W2Container:      document.getElementById(`${prefix}-W2`)
  };

  function regenerate() {
    const seed = parseInt(refs.seedInput.value, 10) || 0;

    const model = randomModel({
      inputDim: INPUT_DIM,
      hiddenDim: HIDDEN_DIM,
      outputDim: OUTPUT_DIM,
      seed
    });

    const ntk = ntkFn(model, DATA_POINTS);

    const labels = buildRowColLabels();
    renderNtkMatrix(refs.matrixContainer, ntk, {
      rowLabels: labels,
      colLabels: labels
    });

    refs.W1Container.innerHTML = `\\(${matrixToTeX('W_1', model.W[0])}\\)`;
    refs.W2Container.innerHTML = `\\(${matrixToTeX('W_2', model.W[1])}\\)`;
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([refs.W1Container, refs.W2Container]).catch(() => {});
    }
  }

  refs.generateBtn.addEventListener('click', regenerate);
  return regenerate;
}

const w1Generate = makeWidget('w1', ntkMatrix);
const w2Generate = makeWidget('w2', ntkMatrixAnalytic2Layer);

function bootstrap() {
  if (window.mathJaxReady || (window.MathJax && window.MathJax.typesetPromise)) {
    w1Generate();
    w2Generate();
  } else {
    setTimeout(bootstrap, 50);
  }
}
bootstrap();
