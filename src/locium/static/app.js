window.addEventListener("DOMContentLoaded", async () => {
  const renderer = new window.Renderer(document.getElementById("map"));
  const meta = await (await fetch("/api/index")).json();
  const buffer = await (await fetch("/api/vectors")).arrayBuffer();
  window.Knn.load(buffer, meta.drawer_count, meta.vector_dim);
  renderer.setData(meta);
  renderer.draw();
  window.__locium = { renderer, meta };
});
