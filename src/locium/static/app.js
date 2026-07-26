/* Locium viewer interaction layer.

   Wires the renderer (render.js) and the client-side k-NN (knn.js) to the
   server API: click-to-read, the wander through neighbours, search, zoom
   and pan, the theme toggle and tunnel confirmation.

   index.html ships only <canvas id="c"> and the title block (#t) -- this
   file is the only one in scope for this task, so every other piece of UI
   (search box, zoom controls, theme toggle, reading panel, trail, tunnel
   form) is built here in JS. Where style.css already carries rules for an
   id (#q, #qn, #zc, #th, #p, #ph, #pm, #pb, #ps, #x -- ported from the
   approved prototype), this file reuses that same id so the CSS applies
   unchanged. Elements the prototype didn't have (trail, neighbours list,
   tunnel form, banners) get a small stylesheet appended below, built from
   the same theme variables so light/dark stay in sync.

   The map never rearranges itself for search: matches get dimmed in place,
   never refocused or refit, so the spatial memory a user builds up over a
   session stays valid. */
(() => {
  const $ = (id) => document.getElementById(id);

  // Relevance chain: how many hits to thread, and the minimum cosine
  // similarity a hit needs to join it (weaker matches are excluded so the
  // chain doesn't zig-zag through noise).
  const CHAIN_MAX = 16;
  const CHAIN_FLOOR = 0.3;

  /* ---- DOM construction ------------------------------------------------
     Drawer text and preview strings are raw palace content and may contain
     "<", "</script>" etc. (see prototype-castle.py's note on this). Every
     place below that inserts such content uses textContent/.value, never
     innerHTML -- the innerHTML calls here are all static literals with no
     interpolated data. */

  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
      #trail{position:fixed;left:34px;top:198px;z-index:6;max-width:236px;
        font:inherit;letter-spacing:.14em;text-transform:uppercase;opacity:.7}
      #trail button{font:inherit;letter-spacing:inherit;text-transform:inherit;
        background:none;border:none;color:var(--ink);cursor:pointer;padding:0}
      #trail button:hover{color:var(--accent)}
      #p .crumbs{margin-bottom:14px;opacity:.65}
      #p .crumbs button{font:inherit;background:none;border:none;color:var(--ink);
        cursor:pointer;padding:0;text-transform:uppercase;letter-spacing:.1em}
      #p .crumbs button:hover{color:var(--accent)}
      #neighbours{margin-top:4px}
      .neighbour{padding:8px 0;border-bottom:1px solid rgba(var(--line),.16);
        cursor:pointer;font:400 10px/1.6 ui-monospace,Menlo,monospace}
      .neighbour:hover{color:var(--accent)}
      .neighbour small{display:block;opacity:.5;letter-spacing:.08em;
        text-transform:uppercase;margin-top:2px}
      #tunnel-slot{margin-top:16px}
      #tunnel-slot input{width:100%;font:inherit;letter-spacing:.05em;
        background:rgba(var(--knock),.6);border:1px solid rgba(var(--line),.36);
        color:var(--ink);padding:6px 8px;margin-bottom:6px;outline:none;
        text-transform:none}
      #tunnel-slot input:focus{border-color:var(--accent)}
      #tunnel-slot button{font:inherit;letter-spacing:.14em;text-transform:uppercase;
        background:rgba(var(--knock),.94);border:1px solid rgba(var(--line),.36);
        color:var(--ink);padding:5px 10px;cursor:pointer}
      #tunnel-slot button:hover{border-color:var(--accent);color:var(--accent)}
      .banner{background:rgba(var(--accent),.16);border:1px solid rgba(var(--accent),.5);
        color:var(--ink);padding:8px 10px;margin-bottom:10px;font:400 10px/1.6
        ui-monospace,Menlo,monospace;letter-spacing:.03em;text-transform:none}
      #banner-slot{position:fixed;left:50%;top:24px;transform:translateX(-50%);
        z-index:8;max-width:420px;width:90%}
    `;
    document.head.appendChild(style);
  }

  function buildUI() {
    injectStyle();

    const search = document.createElement("input");
    search.id = "q";
    search.placeholder = "Search drawers…";
    search.spellcheck = false;
    document.body.appendChild(search);

    const searchCount = document.createElement("div");
    searchCount.id = "qn";
    searchCount.className = "c";
    document.body.appendChild(searchCount);

    const zoomControls = document.createElement("div");
    zoomControls.id = "zc";
    const zoomIn = document.createElement("button");
    zoomIn.type = "button";
    zoomIn.dataset.z = "1.4";
    zoomIn.textContent = "Zoom +";
    const zoomOut = document.createElement("button");
    zoomOut.type = "button";
    zoomOut.dataset.z = "0.72";
    zoomOut.textContent = "Zoom −";
    const fitBtn = document.createElement("button");
    fitBtn.type = "button";
    fitBtn.dataset.z = "0";
    fitBtn.textContent = "Fit";
    const themeBtn = document.createElement("button");
    themeBtn.type = "button";
    themeBtn.id = "th";
    themeBtn.textContent = "Dark";
    zoomControls.append(zoomIn, zoomOut, fitBtn, themeBtn);
    document.body.appendChild(zoomControls);

    const banners = document.createElement("div");
    banners.id = "banner-slot";
    document.body.appendChild(banners);

    const panel = document.createElement("div");
    panel.id = "p";
    // Static skeleton only -- no interpolated data.
    panel.innerHTML =
      '<span id="x">✕</span>' +
      '<div class="crumbs" id="trail"></div>' +
      '<h3 id="ph"></h3>' +
      '<div class="meta" id="pm"></div>' +
      '<div class="rule"></div>' +
      '<div class="body" id="pb"></div>' +
      '<div class="src" id="ps"></div>' +
      '<div class="rule"></div>' +
      '<div id="neighbours"></div>' +
      '<div id="tunnel-slot"></div>';
    document.body.appendChild(panel);
  }

  buildUI();

  const canvas = $("c");
  const renderer = new window.Renderer(canvas);

  const state = {
    meta: null,
    tunnels: [],
    selected: null,
    trail: [],
  };

  /* ---- boot -------------------------------------------------------- */

  async function boot() {
    state.meta = await (await fetch("/api/index")).json();
    const buffer = await (await fetch("/api/vectors")).arrayBuffer();
    window.Knn.load(buffer, state.meta.drawer_count, state.meta.vector_dim);
    renderer.setData(state.meta);
    initTheme();
    await loadTunnels();
    if (state.meta.stale) showBanner("Palace has changed since this index was built.");
    renderer.draw();
  }

  function showBanner(message) {
    const div = document.createElement("div");
    div.className = "banner";
    div.textContent = message;
    $("banner-slot").appendChild(div);
    setTimeout(() => div.remove(), 6000);
  }

  async function loadTunnels() {
    const body = await (await fetch("/api/tunnels")).json();
    state.tunnels = body.tunnels || [];
  }

  /* ---- selection ----------------------------------------------------
     Knn is indexed by a drawer's position in meta.drawers, the same space
     renderer.hitTest returns into -- see the module docstring in knn.js and
     the task brief. A neighbour whose index has no meta.drawers entry is
     dropped defensively (with a console warning) rather than crashing the
     panel; it should never happen when the index was built correctly. */

  function drawerAt(index) {
    return state.meta.drawers[index];
  }

  async function select(index) {
    const drawer = drawerAt(index);
    if (!drawer) {
      console.warn(`locium: select(${index}) has no matching drawer`);
      return;
    }

    state.selected = index;
    renderer.selected = index;

    const record = await (await fetch(`/api/drawer/${drawer.id}`)).json();

    const neighbours = window.Knn.topK(window.Knn.vectorAt(index), 10, index).filter(
      (n) => drawerAt(n.index) !== undefined
    );

    renderer.highlighted = new Set([index, ...neighbours.map((n) => n.index)]);
    renderer.dimmed = new Set();
    renderer.chain = []; // the relevance chain belongs to search, not the wander

    $("ph").textContent = `${record.wing} / ${record.hall} / ${record.room}`;
    $("pm").textContent = record.date || "undated";
    $("pb").textContent = record.text || "(empty drawer)";
    $("ps").textContent = record.source_file ? `source · ${record.source_file}` : "";
    $("p").classList.add("on");

    renderNeighbours(neighbours);
    renderTunnelPanel(record, neighbours[0]);
    pushTrail(`${drawer.wing}/${drawer.room}`, index);
    renderer.draw();
  }

  function closePanel() {
    state.selected = null;
    renderer.selected = null;
    renderer.highlighted = new Set();
    $("p").classList.remove("on");
    renderer.draw();
  }

  function renderNeighbours(neighbours) {
    const list = $("neighbours");
    list.replaceChildren();
    neighbours.forEach((n) => {
      const d = drawerAt(n.index);
      const card = document.createElement("div");
      card.className = "neighbour";
      card.textContent = d.preview || "";
      const small = document.createElement("small");
      small.textContent = `${d.wing}/${d.room} · ${n.distance.toFixed(3)}`;
      card.appendChild(small);
      card.addEventListener("click", () => select(n.index));
      list.appendChild(card);
    });
  }

  /* ---- trail ----------------------------------------------------------
     A repeated click on the same drawer must not grow the trail; only a
     genuine change in selection is appended. */

  function pushTrail(label, index) {
    const last = state.trail[state.trail.length - 1];
    if (last && last.index === index) return;
    state.trail.push({ label, index });

    const crumbs = $("trail");
    crumbs.replaceChildren();
    state.trail.forEach((step, i) => {
      if (i > 0) crumbs.appendChild(document.createTextNode(" › "));
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = step.label;
      btn.addEventListener("click", () => {
        state.trail = state.trail.slice(0, i);
        select(step.index);
      });
      crumbs.appendChild(btn);
    });
  }

  /* ---- tunnels --------------------------------------------------------
     Tunnel identity is the sorted (wing, room) pair; confirming a pair that
     already has a tunnel updates that record instead of creating a new one
     (see tunnels.py), so the panel warns when that is about to happen. */

  function existingTunnel(from, to) {
    const key = [`${from.wing}/${from.room}`, `${to.wing}/${to.room}`].sort().join("|");
    return state.tunnels.find((t) => {
      const stored = [`${t.source.wing}/${t.source.room}`, `${t.target.wing}/${t.target.room}`]
        .sort()
        .join("|");
      return stored === key;
    });
  }

  function renderTunnelPanel(record, nearest) {
    const slot = $("tunnel-slot");
    slot.replaceChildren();
    if (!nearest) return;

    const to = drawerAt(nearest.index);
    const from = { wing: record.wing, room: record.room, id: record.id };
    const already = existingTunnel(from, to);

    if (already) {
      const warning = document.createElement("div");
      warning.className = "banner";
      warning.textContent = `This room pair already has a tunnel ("${already.label || already.id}"). Confirming updates that record.`;
      slot.appendChild(warning);
    }

    const label = document.createElement("input");
    label.id = "tunnel-label";
    label.value = `${from.wing}/${from.room} ↔ ${to.wing}/${to.room}`;
    slot.appendChild(label);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = already ? "Update tunnel" : "Confirm tunnel";
    button.addEventListener("click", () => confirmTunnel(from, to));
    slot.appendChild(button);
  }

  async function confirmTunnel(from, to) {
    await fetch("/api/tunnel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_wing: from.wing,
        source_room: from.room,
        target_wing: to.wing,
        target_room: to.room,
        label: $("tunnel-label") ? $("tunnel-label").value : "",
        source_drawer_id: from.id,
        target_drawer_id: to.id,
      }),
    });
    await loadTunnels();
    showBanner("Tunnel saved.");
    if (state.selected !== null) {
      const record = { wing: from.wing, room: from.room, id: from.id };
      const neighbours = window.Knn.topK(window.Knn.vectorAt(state.selected), 10, state.selected).filter(
        (n) => drawerAt(n.index) !== undefined
      );
      renderTunnelPanel(record, neighbours[0]);
    }
  }

  /* ---- search -----------------------------------------------------------
     Combines a server-embedded semantic score with a literal substring
     match against wing/hall/room names. The view itself never moves. */

  async function search(query) {
    const trimmed = query.trim();
    if (!trimmed) {
      clearSearch();
      return;
    }

    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: trimmed }),
    });
    const { vector } = await response.json();
    const scores = window.Knn.similarities(Float32Array.from(vector));

    const drawers = state.meta.drawers;
    // Rank every drawn drawer by relevance (cosine similarity), strongest first.
    const ranked = drawers
      .map((_, i) => ({ index: i, rel: scores[i] }))
      .sort((a, b) => b.rel - a.rel);
    const semantic = ranked.slice(0, 40).map((r) => r.index);

    const needle = trimmed.toLowerCase();
    const matches = new Set(semantic);
    drawers.forEach((d, i) => {
      if (
        d.wing.toLowerCase().includes(needle) ||
        d.hall.toLowerCase().includes(needle) ||
        d.room.toLowerCase().includes(needle)
      ) {
        matches.add(i);
      }
    });

    // The relevance chain: a path through the strongest genuine hits, in rank
    // order. Weak noise is excluded by the floor; the renderer dots-out any
    // remaining segment below its own threshold.
    renderer.chain = ranked.filter((r) => r.rel >= CHAIN_FLOOR).slice(0, CHAIN_MAX);
    renderer.highlighted = matches;
    renderer.dimmed = new Set(drawers.map((_, i) => i).filter((i) => !matches.has(i)));
    $("qn").textContent = `${matches.size} of ${drawers.length} drawers`;
    $("qn").style.opacity = 1;
    renderer.draw();
  }

  function clearSearch() {
    renderer.dimmed = new Set();
    renderer.highlighted = new Set();
    renderer.chain = [];
    $("q").value = "";
    $("qn").style.opacity = 0;
    renderer.draw();
  }

  /* ---- theme ------------------------------------------------------------ */

  function setTheme(name) {
    if (name !== "light" && name !== "dark") return;
    renderer.setTheme(name);
    document.documentElement.setAttribute("data-theme", name);
    $("th").textContent = name === "dark" ? "Light" : "Dark";
    try {
      localStorage.setItem("locium-theme", name);
    } catch (e) {
      /* localStorage unavailable (private mode, etc.) -- theme just won't persist */
    }
  }

  function initTheme() {
    try {
      const saved = localStorage.getItem("locium-theme");
      if (saved === "dark" || saved === "light") {
        setTheme(saved);
        return;
      }
    } catch (e) {
      /* fall through to system preference */
    }
    setTheme(matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }

  /* ---- zoom / pan / click ------------------------------------------------ */

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      renderer.zoomBy(event.deltaY < 0 ? 1.14 : 1 / 1.14, event.clientX - rect.left, event.clientY - rect.top);
      renderer.draw();
    },
    { passive: false }
  );

  let dragging = false;
  let moved = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    moved = false;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.classList.add("drag");
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    renderer.panBy(dx, dy);
    lastX = event.clientX;
    lastY = event.clientY;
    renderer.draw();
  });

  window.addEventListener("pointerup", (event) => {
    if (!dragging) return;
    dragging = false;
    canvas.classList.remove("drag");
    if (moved) return; // a drag must not be treated as a click
    const rect = canvas.getBoundingClientRect();
    const hit = renderer.hitTest(event.clientX - rect.left, event.clientY - rect.top);
    if (hit !== null) select(hit);
  });

  $("zc").addEventListener("click", (event) => {
    const z = event.target.dataset.z;
    if (z === undefined) return;
    if (Number(z) === 0) {
      renderer.home();
      renderer.draw();
      return;
    }
    renderer.zoomBy(Number(z), renderer.viewW / 2, renderer.viewH / 2);
    renderer.draw();
  });

  $("th").addEventListener("click", () => setTheme(renderer.themeName === "dark" ? "light" : "dark"));

  $("q").addEventListener("keydown", (event) => {
    if (event.key === "Enter") search(event.target.value);
    if (event.key === "Escape") clearSearch();
  });

  $("x").addEventListener("click", closePanel);

  window.__locium = { renderer, state, select, search, setTheme, confirmTunnel };
  boot();
})();
