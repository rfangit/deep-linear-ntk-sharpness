// ============================================================================
// INCREMENTAL CACHE - Efficient downsampling + EMA without O(n) recomputation
// ============================================================================

export class IncrementalCache {
  constructor(emaWindow, maxPlotPoints, valueKeys, initEmaValues) {
    this.emaWindow = emaWindow;
    this.maxPlotPoints = maxPlotPoints;
    this.valueKeys = Array.isArray(valueKeys) ? valueKeys : [valueKeys];
    this.initEmaValues = initEmaValues || {};

    this.stride = 1;
    this.downsampledRaw = [];
    this.downsampledSmoothed = [];

    this.emaData = [];
    this.lastEmaValues = {};
    for (const key of this.valueKeys) {
      this.lastEmaValues[key] = this.initEmaValues[key] !== undefined
        ? this.initEmaValues[key]
        : 0;
    }

    this.rollingMax = {};
    for (const key of this.valueKeys) this.rollingMax[key] = -Infinity;

    this.lastProcessedIndex = -1;
  }

  update(fullData) {
    if (fullData.length === 0) {
      return { downsampledRaw: [], downsampledSmoothed: [], max: this.rollingMax };
    }

    const newStride = this._computeStride(fullData.length);
    const needsRebuild = newStride !== this.stride ||
                        this.lastProcessedIndex >= fullData.length ||
                        this.lastProcessedIndex < 0;

    if (needsRebuild) return this._rebuildCache(fullData, newStride);

    for (let i = this.lastProcessedIndex + 1; i < fullData.length; i++) {
      const point = fullData[i];
      const emaPoint = { iteration: point.iteration };
      for (const key of this.valueKeys) {
        const alpha = this.emaWindow === 1 ? 1 : 1 / this.emaWindow;
        const smoothed = alpha * point[key] + (1 - alpha) * this.lastEmaValues[key];
        emaPoint[key] = smoothed;
        this.lastEmaValues[key] = smoothed;
        if (smoothed > this.rollingMax[key]) this.rollingMax[key] = smoothed;
      }
      this.emaData.push(emaPoint);

      if (i % this.stride === 0) {
        this.downsampledRaw.push(point);
        this.downsampledSmoothed.push(emaPoint);
      }
    }

    const lastIndex = fullData.length - 1;
    if (lastIndex % this.stride !== 0) {
      if (this.downsampledRaw.length > 0 &&
          this.downsampledRaw[this.downsampledRaw.length - 1].iteration !== fullData[lastIndex].iteration) {
        const lastCachedIteration = this.downsampledRaw[this.downsampledRaw.length - 1].iteration;
        const secondToLastStrideIndex = Math.floor(this.lastProcessedIndex / this.stride) * this.stride;
        if (lastCachedIteration > secondToLastStrideIndex) {
          this.downsampledRaw.pop();
          this.downsampledSmoothed.pop();
        }
      }
      this.downsampledRaw.push(fullData[lastIndex]);
      this.downsampledSmoothed.push(this.emaData[this.emaData.length - 1]);
    }

    this.lastProcessedIndex = fullData.length - 1;
    return {
      downsampledRaw: this.downsampledRaw,
      downsampledSmoothed: this.downsampledSmoothed,
      emaData: this.emaData,
      max: this.rollingMax
    };
  }

  _rebuildCache(fullData, newStride) {
    this.stride = newStride;
    this.downsampledRaw = [];
    this.downsampledSmoothed = [];
    this.emaData = [];

    for (const key of this.valueKeys) {
      this.lastEmaValues[key] = this.initEmaValues[key] !== undefined
        ? this.initEmaValues[key]
        : fullData[0]?.[key] || 0;
      this.rollingMax[key] = -Infinity;
    }

    const alpha = this.emaWindow === 1 ? 1 : 1 / this.emaWindow;
    for (let i = 0; i < fullData.length; i++) {
      const point = fullData[i];
      const emaPoint = { iteration: point.iteration };
      for (const key of this.valueKeys) {
        const smoothed = alpha * point[key] + (1 - alpha) * this.lastEmaValues[key];
        emaPoint[key] = smoothed;
        this.lastEmaValues[key] = smoothed;
        if (smoothed > this.rollingMax[key]) this.rollingMax[key] = smoothed;
      }
      this.emaData.push(emaPoint);
      if (i % this.stride === 0) {
        this.downsampledRaw.push(point);
        this.downsampledSmoothed.push(emaPoint);
      }
    }

    const lastIndex = fullData.length - 1;
    if (lastIndex % this.stride !== 0 && lastIndex >= 0) {
      this.downsampledRaw.push(fullData[lastIndex]);
      this.downsampledSmoothed.push(this.emaData[lastIndex]);
    }

    this.lastProcessedIndex = fullData.length - 1;
    return {
      downsampledRaw: this.downsampledRaw,
      downsampledSmoothed: this.downsampledSmoothed,
      emaData: this.emaData,
      max: this.rollingMax
    };
  }

  _computeStride(dataLength) {
    if (dataLength <= this.maxPlotPoints) return 1;
    let stride = 1;
    while (dataLength / stride > this.maxPlotPoints) stride *= 2;
    return stride;
  }

  setEmaWindow(newWindow) {
    if (newWindow !== this.emaWindow) {
      this.emaWindow = newWindow;
      this.lastProcessedIndex = -1;
    }
  }

  clear() {
    this.stride = 1;
    this.downsampledRaw = [];
    this.downsampledSmoothed = [];
    this.emaData = [];
    this.lastProcessedIndex = -1;
    for (const key of this.valueKeys) {
      this.lastEmaValues[key] = this.initEmaValues[key] !== undefined
        ? this.initEmaValues[key]
        : 0;
      this.rollingMax[key] = -Infinity;
    }
  }
}
