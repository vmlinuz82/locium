/* Client-side k-NN over int8 vectors.
   Quantised vectors are dequantised once into a flat Float32Array; scoring
   5k x 384 is a few milliseconds, so selection needs no server round-trip. */
window.Knn = (() => {
  const SCALE = 127.0;
  let vectors = null;
  let count = 0;
  let dim = 0;

  function load(buffer, drawerCount, dimensions) {
    const raw = new Int8Array(buffer);
    count = drawerCount;
    dim = dimensions;
    vectors = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) vectors[i] = raw[i] / SCALE;
  }

  function vectorAt(index) {
    return vectors.subarray(index * dim, (index + 1) * dim);
  }

  function similarities(query) {
    const scores = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      let dot = 0;
      const offset = i * dim;
      for (let d = 0; d < dim; d += 1) dot += vectors[offset + d] * query[d];
      scores[i] = dot;
    }
    return scores;
  }

  function topK(query, k, excludeIndex = -1) {
    const scores = similarities(query);
    const order = [];
    for (let i = 0; i < count; i += 1) {
      if (i === excludeIndex) continue;
      order.push({ index: i, distance: 1 - scores[i] });
    }
    order.sort((a, b) => a.distance - b.distance);
    return order.slice(0, k);
  }

  /* An independent scoring store over a second int8 matrix -- used for the
     whole-exchange vectors that back the recall-gap verdict, so they can
     never be confused with (or overwrite) the drawer vectors above. */
  function makeStore(buffer, storeCount, storeDim) {
    const raw = new Int8Array(buffer);
    const data = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) data[i] = raw[i] / SCALE;
    return {
      count: storeCount,
      similarities(query) {
        const scores = new Float32Array(storeCount);
        for (let i = 0; i < storeCount; i += 1) {
          let dot = 0;
          const offset = i * storeDim;
          for (let d = 0; d < storeDim; d += 1) dot += data[offset + d] * query[d];
          scores[i] = dot;
        }
        return scores;
      },
    };
  }

  return { load, topK, similarities, vectorAt, makeStore, get count() { return count; } };
})();
