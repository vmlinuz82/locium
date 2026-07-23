/* Canvas 2D renderer.
   SVG would mean one DOM node per drawer — fine at 500, dead at 50k. The
   palace grows daily, so everything here is immediate-mode drawing. */
window.Renderer = class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.meta = { drawers: [], wings: [], clusters: [], arcs: [] };
    this.colourMode = "room";
    this.dimmed = new Set();
    this.highlighted = new Set();
    this.activeArcs = [];
    this.selected = null;
    this.roomColours = new Map();
    this._resize();
    window.addEventListener("resize", () => { this._resize(); this.draw(); });
  }

  _resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * ratio;
    this.canvas.height = rect.height * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.viewW = rect.width;
    this.viewH = rect.height;
  }

  setData(meta) {
    this.meta = meta;
    const rooms = [...new Set(meta.drawers.map((d) => d.room))].sort();
    rooms.forEach((room, i) => {
      this.roomColours.set(room, `hsl(${(i * 360) / rooms.length}, 62%, 62%)`);
    });
    this.fit();
  }

  fit() {
    this.scale = Math.min(this.viewW / 1000, this.viewH / 1000) * 0.92;
    this.offsetX = (this.viewW - 1000 * this.scale) / 2;
    this.offsetY = (this.viewH - 1000 * this.scale) / 2;
  }

  zoomLevel() {
    if (this.scale < 0.55) return "far";
    if (this.scale < 2.2) return "mid";
    return "near";
  }

  worldToScreen(x, y) {
    return [x * this.scale + this.offsetX, y * this.scale + this.offsetY];
  }

  screenToWorld(px, py) {
    return [(px - this.offsetX) / this.scale, (py - this.offsetY) / this.scale];
  }

  zoomBy(factor, px, py) {
    const [wx, wy] = this.screenToWorld(px, py);
    this.scale = Math.max(0.2, Math.min(30, this.scale * factor));
    this.offsetX = px - wx * this.scale;
    this.offsetY = py - wy * this.scale;
  }

  panBy(dx, dy) {
    this.offsetX += dx;
    this.offsetY += dy;
  }

  focusOn(x, y) {
    this.scale = Math.max(this.scale, 4);
    this.offsetX = this.viewW / 2 - x * this.scale;
    this.offsetY = this.viewH / 2 - y * this.scale;
  }

  hitTest(px, py) {
    const [wx, wy] = this.screenToWorld(px, py);
    const radius = 6 / this.scale;
    let best = null;
    let bestDistance = radius;
    this.meta.drawers.forEach((drawer, index) => {
      const distance = Math.hypot(drawer.x - wx, drawer.y - wy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  }

  _drawerColour(drawer) {
    if (this.colourMode === "age") {
      const year = Number((drawer.date || "").slice(0, 4)) || 2026;
      const month = Number((drawer.date || "").slice(5, 7)) || 1;
      const age = (2027 - year) * 12 - month;
      const light = Math.max(28, 72 - age * 3);
      return `hsl(28, 70%, ${light}%)`;
    }
    return this.roomColours.get(drawer.room) || "#8b8fa3";
  }

  draw() {
    const ctx = this.ctx;
    ctx.fillStyle = "#12131a";
    ctx.fillRect(0, 0, this.viewW, this.viewH);

    this._drawWings(ctx);
    this._drawArcs(ctx);
    this._drawDrawers(ctx);
    if (this.zoomLevel() !== "far") this._drawClusterLabels(ctx);
  }

  _drawWings(ctx) {
    ctx.lineWidth = 1;
    this.meta.wings.forEach((wing) => {
      const [x, y] = this.worldToScreen(wing.rect[0], wing.rect[1]);
      const w = wing.rect[2] * this.scale;
      const h = wing.rect[3] * this.scale;
      ctx.strokeStyle = "#2c2f3d";
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "#6b7089";
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(`${wing.name} · ${wing.count}`, x + 6, y + 16);
    });
  }

  _drawArcs(ctx) {
    if (!this.activeArcs.length) return;
    ctx.strokeStyle = "rgba(122,162,247,0.5)";
    ctx.lineWidth = 1;
    this.activeArcs.forEach(([a, b]) => {
      const from = this.meta.drawers[a];
      const to = this.meta.drawers[b];
      if (!from || !to) return;
      const [x1, y1] = this.worldToScreen(from.x, from.y);
      const [x2, y2] = this.worldToScreen(to.x, to.y);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });
  }

  _drawDrawers(ctx) {
    const level = this.zoomLevel();
    const radius = level === "far" ? 1.2 : level === "mid" ? 2.4 : 4;
    this.meta.drawers.forEach((drawer, index) => {
      const [x, y] = this.worldToScreen(drawer.x, drawer.y);
      if (x < -20 || y < -20 || x > this.viewW + 20 || y > this.viewH + 20) return;

      const isDimmed = this.dimmed.has(index);
      ctx.globalAlpha = isDimmed ? 0.08 : this.highlighted.has(index) ? 1 : 0.75;
      ctx.fillStyle = this._drawerColour(drawer);
      ctx.beginPath();
      ctx.arc(x, y, index === this.selected ? radius + 3 : radius, 0, Math.PI * 2);
      ctx.fill();

      if (index === this.selected) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }

  _drawClusterLabels(ctx) {
    ctx.fillStyle = "#c3c7d6";
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    this.meta.clusters.forEach((cluster) => {
      const [x, y] = this.worldToScreen(cluster.centroid[0], cluster.centroid[1]);
      if (x < 0 || y < 0 || x > this.viewW || y > this.viewH) return;
      ctx.fillText(cluster.label, x, y - 8);
    });
    ctx.textAlign = "left";
  }
};
