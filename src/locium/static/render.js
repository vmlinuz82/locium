/* Canvas 2D renderer — draws the palace as an architect's floorplan.
   SVG would mean one DOM node per drawer; fine at 500, dead at 6,000+, so
   everything here is immediate-mode drawing, redone every frame.

   Ported from the approved prototype at
   .superpowers/sdd/prototype-castle.py (the `HTML` string). That prototype
   is the source of truth for every visual constant below — this file only
   replaces its module-level globals with instance state and its embedded
   `D` payload with `setData(meta)`. */

/* Two prints of the same drawing: ink on paper, and its negative — light
   linework on a dark ground, the way a plan reads backlit. */
const THEMES = {
  light: {
    paper: "#e9e7e1", line: "21,21,15", knock: "233,231,225", accent: "168,51,26",
    grain: 8, dot0: 172, dotSpan: -158, wing: 0.86, hall: 0.42, room: 0.34, dim: 0.10,
  },
  dark: {
    paper: "#15171b", line: "214,209,196", knock: "21,23,27", accent: "224,138,92",
    grain: 13, dot0: 74, dotSpan: 150, wing: 0.72, hall: 0.34, room: 0.26, dim: 0.09,
  },
};

/* Zoom limits, as multiples of the fit-to-screen scale. The ceiling is set by
   the densest chambers, not by taste: drawer coordinates are quantised at
   build time, and a saturated chamber packs its dots roughly 0.1 world units
   apart. Since dot radius is capped at 4.2px (see draw()), the dots stay wider
   than the gaps between them until scale passes ~84 -- about 105x baseScale on
   a 900px-tall viewport. 250 clears that with room to read and click. */
const ZOOM_MIN = 0.55;
const ZOOM_MAX = 250;

function boxesOverlap(a, b) {
  return !(a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y);
}

window.Renderer = class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    this.meta = { wings: [], halls: [], chambers: [], drawers: [], arcs: [], drawer_count: 0, vector_dim: 0 };
    this._t = []; // per-drawer recency, 0 (oldest) .. 1 (newest), aligned to meta.drawers

    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.baseScale = 1;
    this.viewW = 0;
    this.viewH = 0;

    this.dimmed = new Set();
    this.highlighted = new Set();
    // Ordered relevance chain drawn between search hits, strongest first.
    // Each entry: { index, rel } where rel is cosine similarity 0..1.
    this.chain = [];
    // Neighbour star: rays from a clicked drawer to its nearest neighbours.
    // Each entry: { from, to, rel }.
    this.rays = [];
    this.selected = null;
    // Drawer index under the cursor in the results list, marked on the
    // floor so a card can be matched to its dot at a glance.
    this.hovered = null;

    this.themeName = "light";
    this.theme = THEMES.light;

    this._taken = []; // label collision boxes, rebuilt every draw()

    this.fit();
    this.home();

    window.addEventListener("resize", () => {
      const ratio = this.baseScale > 0 ? this.scale / this.baseScale : 1;
      this.fit();
      this.scale = this.baseScale * ratio;
      this.draw();
    });
  }

  setData(meta) {
    this.meta = {
      wings: meta.wings || [],
      halls: meta.halls || [],
      chambers: meta.chambers || [],
      drawers: meta.drawers || [],
      arcs: meta.arcs || [],
      drawer_count: meta.drawer_count || 0,
      vector_dim: meta.vector_dim || 0,
    };
    this._t = this._computeRecency(this.meta.drawers);
    this.fit();
    this.home();
  }

  _computeRecency(drawers) {
    const seen = new Set();
    const days = [];
    drawers.forEach((d) => {
      const day = (d.date || "").slice(0, 10);
      if (day && !seen.has(day)) {
        seen.add(day);
        days.push(day);
      }
    });
    days.sort();
    const span = Math.max(days.length - 1, 1);
    const index = new Map(days.map((day, i) => [day, i / span]));
    return drawers.map((d) => index.get((d.date || "").slice(0, 10)) ?? 0);
  }

  /* Recompute the backing store for the current viewport and DPR, and the
     base scale that fits the 1000x1000 world into it. Does not touch the
     current pan/zoom — call home() for that. */
  fit() {
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.viewW = rect.width;
    this.viewH = rect.height;
    this.canvas.width = rect.width * ratio;
    this.canvas.height = rect.height * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.baseScale = Math.min(this.viewW / 1000, this.viewH / 1000) * 0.885;
  }

  /* Reset pan/zoom to show the whole building, centred. */
  home() {
    this.scale = this.baseScale;
    this.offsetX = (this.viewW - 1000 * this.scale) / 2;
    this.offsetY = (this.viewH - 1000 * this.scale) / 2;
  }

  worldToScreen(x, y) {
    return [x * this.scale + this.offsetX, y * this.scale + this.offsetY];
  }

  screenToWorld(px, py) {
    return [(px - this.offsetX) / this.scale, (py - this.offsetY) / this.scale];
  }

  zoomBy(factor, px, py) {
    const [wx, wy] = this.screenToWorld(px, py);
    this.scale = Math.max(this.baseScale * ZOOM_MIN, Math.min(this.baseScale * ZOOM_MAX, this.scale * factor));
    this.offsetX = px - wx * this.scale;
    this.offsetY = py - wy * this.scale;
  }

  panBy(dx, dy) {
    this.offsetX += dx;
    this.offsetY += dy;
  }

  focusOn(x, y) {
    this.scale = Math.min(this.baseScale * ZOOM_MAX, Math.max(this.scale, this.baseScale * 4));
    this.offsetX = this.viewW / 2 - x * this.scale;
    this.offsetY = this.viewH / 2 - y * this.scale;
  }

  /* Animated pan that centres a point at the CURRENT zoom -- the zoom level
     is the user's choice and a flight must not override it. Eases over
     ~350ms so the eye can track WHERE the map is going; a new flight (or
     stopFlight from a manual pan/zoom) cancels the previous one mid-air. */
  flyTo(x, y, duration = 350) {
    const targetX = this.viewW / 2 - x * this.scale;
    const targetY = this.viewH / 2 - y * this.scale;
    const from = { x: this.offsetX, y: this.offsetY };

    this.stopFlight();
    const started = performance.now();
    const easeOut = (t) => 1 - Math.pow(1 - t, 3);
    const step = (now) => {
      const t = Math.min((now - started) / duration, 1);
      const k = easeOut(t);
      this.offsetX = from.x + (targetX - from.x) * k;
      this.offsetY = from.y + (targetY - from.y) * k;
      this.draw();
      if (t < 1) this._flight = requestAnimationFrame(step);
    };
    this._flight = requestAnimationFrame(step);
  }

  stopFlight() {
    if (this._flight) cancelAnimationFrame(this._flight);
    this._flight = null;
  }

  hitTest(px, py) {
    const [wx, wy] = this.screenToWorld(px, py);
    const tol = 8 / this.scale;
    let best = null;
    let bestDistance = tol;
    this.meta.drawers.forEach((d, i) => {
      const distance = Math.hypot(d.x - wx, d.y - wy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    });
    return best;
  }

  setTheme(name) {
    if (!THEMES[name]) return;
    this.themeName = name;
    this.theme = THEMES[name];
    this.draw();
  }

  /* Per-pixel noise overlay giving the paper its grain. Sized off the
     canvas's actual backing-store pixels (not CSS pixels) — putImageData
     ignores the current transform, so on a high-DPI screen using the CSS
     size here would only grain the top-left quarter of the sheet. */
  _grain() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const g = this.ctx.createImageData(w, h);
    const p = g.data;
    const alpha = this.theme.grain;
    for (let i = 0; i < p.length; i += 4) {
      const n = (Math.random() * 255) | 0;
      p[i] = p[i + 1] = p[i + 2] = n;
      p[i + 3] = alpha;
    }
    return g;
  }

  /* The grain, cached on an offscreen canvas. Regenerating noise per frame
     was fine when draws only happened on interaction, but the hover flight
     redraws every animation frame — and a full-screen ImageData allocation
     plus one Math.random() per pixel per frame freezes the tab on large
     high-DPI viewports. The noise is static; only a resize or a theme
     change (different grain alpha) warrants rolling it again. */
  _grainLayer() {
    const key = `${this.canvas.width}x${this.canvas.height}:${this.theme.grain}`;
    if (this._grainKey !== key) {
      const layer = document.createElement("canvas");
      layer.width = this.canvas.width;
      layer.height = this.canvas.height;
      layer.getContext("2d").putImageData(this._grain(), 0, 0);
      this._grainCanvas = layer;
      this._grainKey = key;
    }
    return this._grainCanvas;
  }

  /* A wall drawn as a filled band between its outer and inner edge, not a
     stroke — this is what makes it read as masonry (poché) rather than an
     outline. */
  _poche(rect, thickness, tone) {
    const ctx = this.ctx;
    const [sx, sy] = this.worldToScreen(rect[0], rect[1]);
    const sw = rect[2] * this.scale;
    const sh = rect[3] * this.scale;
    if (sw < 3 || sh < 3) return;
    const tt = Math.min(thickness, sw / 2.6, sh / 2.6);
    ctx.fillStyle = tone;
    ctx.beginPath();
    ctx.rect(sx, sy, sw, sh);
    ctx.rect(sx + tt, sy + tt, Math.max(sw - 2 * tt, 0), Math.max(sh - 2 * tt, 0));
    ctx.fill("evenodd");
  }

  /* A label with a knockout rect behind it, nudged down (up to three times)
     to avoid the labels already placed this frame, and dropped rather than
     painted over a neighbour. */
  _tag(text, rect, size, alpha, track, accent) {
    const ctx = this.ctx;
    const TH = this.theme;
    const [sx, sy] = this.worldToScreen(rect[0], rect[1]);
    const sw = rect[2] * this.scale;
    const sh = rect[3] * this.scale;
    ctx.font = `400 ${size}px ui-monospace,Menlo,monospace`;
    ctx.letterSpacing = `${track}px`;
    let t = text.toUpperCase();
    let m = ctx.measureText(t).width;
    const av = sw - 9;
    if (m > av) {
      const k = Math.floor((t.length * av) / m);
      if (k < 3) return;
      t = t.slice(0, k);
      m = ctx.measureText(t).width;
    }
    if (sh < size + 7) return;

    const aw = accent ? ctx.measureText(accent).width + 6 : 0;
    const bw = m + aw + 4;
    const bh = size + 2.5;
    let bx = sx + 4.5;
    let by = sy + 4.5;
    const box = { x: bx - 2, y: by - 0.5, w: bw, h: bh };
    let n = 0;
    while (this._taken.some((o) => boxesOverlap(box, o)) && n < 3) {
      by += bh + 1.5;
      box.y = by - 0.5;
      n += 1;
    }
    if (this._taken.some((o) => boxesOverlap(box, o)) || by + bh > sy + sh - 1.5) return;
    this._taken.push(box);

    ctx.fillStyle = `rgba(${TH.knock},0.9)`;
    ctx.fillRect(box.x, box.y, bw, bh);
    ctx.fillStyle = `rgba(${TH.line},${alpha})`;
    ctx.textBaseline = "top";
    ctx.fillText(t, bx, by);
    if (accent) {
      ctx.fillStyle = `rgba(${TH.accent},0.92)`;
      ctx.fillText(accent, bx + m + 5, by);
    }
  }

  /* One relevance-weighted line between two drawers: warm+solid when strong,
     cool+dotted below DOTTED_BELOW. Shared by the search chain and the
     neighbour star so both read as the same gradient. */
  _drawSegment(ai, bi, rel) {
    const a = this.meta.drawers[ai];
    const b = this.meta.drawers[bi];
    if (!a || !b) return;
    const ctx = this.ctx;
    const accent = this.theme.accent.split(",").map(Number);
    const weak = this.themeName === "dark" ? [122, 132, 158] : [110, 120, 146];
    const LO = 0.35, HI = 0.7; // relevance range mapped onto the colour ramp
    const t = Math.max(0, Math.min(1, (rel - LO) / (HI - LO)));
    const c = accent.map((hi, i) => Math.round(weak[i] + (hi - weak[i]) * t));
    const [x1, y1] = this.worldToScreen(a.x, a.y);
    const [x2, y2] = this.worldToScreen(b.x, b.y);

    ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${(0.3 + 0.55 * t).toFixed(3)})`;
    ctx.lineWidth = 1.0 + 1.7 * t;
    ctx.setLineDash(rel < 0.5 ? [3, 3.5] : []);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  /* Search: a path through the ranked hits, strongest to weakest, each
     segment weighted by its weaker (later) endpoint. Neighbours (star): rays
     from the clicked drawer to each of its nearest neighbours. */
  _drawChain() {
    const ctx = this.ctx;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    for (let k = 0; k < this.chain.length - 1; k += 1) {
      const rel = Math.min(this.chain[k].rel, this.chain[k + 1].rel);
      this._drawSegment(this.chain[k].index, this.chain[k + 1].index, rel);
    }
    for (const ray of this.rays) {
      this._drawSegment(ray.from, ray.to, ray.rel);
    }
    ctx.setLineDash([]);
  }

  draw() {
    const ctx = this.ctx;
    const TH = this.theme;

    ctx.fillStyle = TH.paper;
    ctx.fillRect(0, 0, this.viewW, this.viewH);
    // drawImage honours the DPR transform (putImageData did not), so paint
    // the backing-store-sized grain layer with the transform reset.
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this._grainLayer(), 0, 0);
    ctx.restore();

    ctx.strokeStyle = `rgba(${TH.line},0.32)`;
    ctx.lineWidth = 1;
    ctx.strokeRect(18.5, 18.5, this.viewW - 37, this.viewH - 37);
    ctx.strokeStyle = `rgba(${TH.line},0.12)`;
    ctx.strokeRect(23.5, 23.5, this.viewW - 47, this.viewH - 47);
    ctx.textAlign = "left";

    this.meta.wings.forEach((g) => this._poche(g.rect, 3.0, `rgba(${TH.line},${TH.wing})`));
    this.meta.halls.forEach((g) => this._poche(g.rect, 1.5, `rgba(${TH.line},${TH.hall})`));
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = `rgba(${TH.line},${TH.room})`;
    this.meta.chambers.forEach((c) => {
      const [sx, sy] = this.worldToScreen(c.rect[0], c.rect[1]);
      ctx.strokeRect(sx + 0.25, sy + 0.25, c.rect[2] * this.scale - 0.5, c.rect[3] * this.scale - 0.5);
    });

    // Cluster zones: recovered topics inside saturated chambers. Dashed, not
    // walled -- these are soft groupings the build computed, not rooms the
    // taxonomy declared, and the linework should say so. Only at deep zoom,
    // where a mega-chamber is what fills the screen.
    if (this.scale > this.baseScale * 2.5) {
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = `rgba(${TH.line},0.2)`;
      ctx.setLineDash([2, 3]);
      this.meta.chambers.forEach((c) => {
        (c.clusters || []).forEach((cl) => {
          const [sx, sy] = this.worldToScreen(cl.rect[0], cl.rect[1]);
          ctx.strokeRect(sx, sy, cl.rect[2] * this.scale, cl.rect[3] * this.scale);
        });
      });
      ctx.setLineDash([]);
    }

    // Relevance chain, under the dots so the accent hits sit on top of it.
    this._drawChain();

    // Radius is capped both ends: floor so a far-zoomed dot stays visible,
    // ceiling so a near-zoomed chamber doesn't turn into overlapping blobs.
    const radius = Math.min(Math.max(0.8, 1.35 * this.scale), 4.2);
    this.meta.drawers.forEach((d, i) => {
      const [x, y] = this.worldToScreen(d.x, d.y);
      if (x < -6 || y < -6 || x > this.viewW + 6 || y > this.viewH + 6) return;

      if (this.highlighted.has(i)) {
        ctx.fillStyle = `rgba(${TH.accent},0.95)`;
        ctx.beginPath();
        ctx.arc(x, y, radius + 1.1, 0, Math.PI * 2);
        ctx.fill();
      } else if (this.dimmed.has(i)) {
        ctx.fillStyle = `rgba(${TH.line},${TH.dim})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      } else {
        const v = Math.round(TH.dot0 + this._t[i] * TH.dotSpan);
        ctx.fillStyle = `rgb(${v},${v},${v - 5})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      if (i === this.selected) {
        ctx.strokeStyle = `rgb(${TH.accent})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, radius + 4.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.lineWidth = 0.5;
      }

    });

    // Labels grow with the zoom they annotate — a fixed pixel size reads as
    // microscopic once a chamber fills the screen. Clamped so they never
    // dominate. Importance order: wings, then halls, then chambers.
    this._taken = [];
    const zf = Math.min(Math.max(this.scale / this.baseScale, 1), 3.4);
    const up = (a) => Math.min(a + (zf - 1) * 0.16, 0.95);

    this.meta.wings.forEach((g) => this._tag(g.name, g.rect, 8.0 * zf, 0.95, 2.2 * zf, String(g.count)));
    if (this.scale > this.baseScale * 0.75) {
      this.meta.halls.forEach((g) => {
        if (g.rect[2] * this.scale > 66) this._tag(g.name, g.rect, 6.6 * zf, up(0.62), 1.4 * zf, null);
      });
    }
    if (this.scale > this.baseScale * 0.95) {
      this.meta.chambers.forEach((c) => {
        if (c.rect[2] * this.scale > 44) {
          this._tag(c.name, c.rect, 5.6 * zf, up(0.42), 0.9 * zf, c.capped ? String(c.count) : null);
        }
      });
    }
    // Cluster labels last, so a chamber's own name wins the collision fight.
    if (this.scale > this.baseScale * 2.5) {
      this.meta.chambers.forEach((c) => {
        (c.clusters || []).forEach((cl) => {
          if (cl.label && cl.rect[2] * this.scale > 60) {
            this._tag(cl.label, cl.rect, 5.2 * zf, up(0.34), 0.7 * zf, String(cl.count));
          }
        });
      });
    }
    ctx.letterSpacing = "0px";

    this._drawHoverMarker();
  }

  /* The hovered result's dot, drawn LAST so nothing -- labels, knockouts,
     denser dots -- can sit on top of it: an accent glow wide enough to spot
     from across the sheet, a filled core, and a target ring. */
  _drawHoverMarker() {
    if (this.hovered === null) return;
    const d = this.meta.drawers[this.hovered];
    if (!d) return;
    const ctx = this.ctx;
    const TH = this.theme;
    const [x, y] = this.worldToScreen(d.x, d.y);
    const radius = Math.min(Math.max(0.8, 1.35 * this.scale), 4.2);

    const glow = ctx.createRadialGradient(x, y, radius, x, y, radius + 26);
    glow.addColorStop(0, `rgba(${TH.accent},0.55)`);
    glow.addColorStop(1, `rgba(${TH.accent},0)`);
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, radius + 26, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = `rgb(${TH.accent})`;
    ctx.beginPath();
    ctx.arc(x, y, radius + 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = `rgb(${TH.accent})`;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(x, y, radius + 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 0.5;
  }
};
