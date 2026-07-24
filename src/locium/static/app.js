/* Minimal stub: wires the renderer to the API and nothing else.
   R7 replaces this file with the full interaction layer (zoom/pan controls,
   search, reading panel, theme toggle). This stub exists only so the
   renderer can be verified by hand until then. */
window.addEventListener("DOMContentLoaded", async () => {
  const renderer = new window.Renderer(document.getElementById("c"));
  const meta = await (await fetch("/api/index")).json();
  const buffer = await (await fetch("/api/vectors")).arrayBuffer();
  window.Knn.load(buffer, meta.drawer_count, meta.vector_dim);
  renderer.setData(meta);
  renderer.draw();
  window.__locium = { renderer, meta };
});
