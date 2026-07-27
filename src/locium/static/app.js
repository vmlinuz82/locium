/* Locium viewer interaction layer.

   Wires the renderer (render.js) and the client-side k-NN (knn.js) to the
   server API: click-to-read, the wander through neighbours, search, zoom
   and pan, and the theme toggle. Nothing here writes to the palace.

   index.html ships only <canvas id="c"> and the title block (#t) -- this
   file is the only one in scope for this task, so every other piece of UI
   (search box, zoom controls, theme toggle, reading panel) is built here in
   JS. Where style.css already carries rules for an id (#q, #qn, #zc, #th, #p,
   #ph, #pm, #pb, #ps, #x, and the scrollbars shared with #mb -- ported from
   the approved prototype), this file reuses that same id so the CSS applies
   unchanged. Elements the prototype didn't have (neighbours list, full-text
   popup, banners) get a small stylesheet appended below, built from the same
   theme variables so light/dark stay in sync.

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

  // What MemPalace would actually hand an agent for this query: its search
  // tool returns the top RECALL_TOP drawers, and the agent's own protocol
  // treats anything under RECALL_FLOOR cosine similarity as "no relevant
  // prior context". The floor doubles as the match cutoff -- without one the
  // top 40 by rank always light up, so a query nothing answers (a ticket id,
  // say) still paints 40 arbitrary dots red and reads as 40 results.
  const RECALL_FLOOR = 0.15;
  const RECALL_TOP = 5;
  const MATCH_MAX = 40;

  /* ---- DOM construction ------------------------------------------------
     Drawer text and preview strings are raw palace content and may contain
     "<", "</script>" etc. (see prototype-castle.py's note on this). Every
     place below that inserts such content uses textContent/.value, never
     innerHTML -- the innerHTML calls here are all static literals with no
     interpolated data. */

  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
      #neighbours{margin-top:4px}
      .neighbour{padding:8px 0;border-bottom:1px solid rgba(var(--line),.16);
        cursor:pointer;font:400 10px/1.6 ui-monospace,Menlo,monospace}
      .neighbour:hover{color:var(--accent)}
      .neighbour .text{display:block;white-space:pre-wrap;word-break:break-word;
        text-transform:none;letter-spacing:0}
      .neighbour small{display:block;opacity:.5;letter-spacing:.08em;
        text-transform:uppercase;margin-top:6px}
      .banner{background:rgba(var(--accent),.16);border:1px solid rgba(var(--accent),.5);
        color:var(--ink);padding:8px 10px;margin-bottom:10px;font:400 10px/1.6
        ui-monospace,Menlo,monospace;letter-spacing:.03em;text-transform:none}
      #banner-slot{position:fixed;left:50%;top:24px;transform:translateX(-50%);
        z-index:8;max-width:420px;width:90%}
      /* The clear affordance sits inside #q's box. #q is fixed at left:34px
         with width:236px, so its right edge is 270px; 242+26 lands the button
         2px inside that. Same font and vertical padding as the input, so both
         boxes come out the same height without hard-coding one. */
      #q{padding-right:30px}
      #qx{position:fixed;left:242px;top:104px;z-index:7;width:26px;
        background:none;border:1px solid transparent;padding:7px 0;
        font:inherit;color:var(--ink);text-align:center;cursor:pointer;
        opacity:0;pointer-events:none;transition:opacity .15s}
      #qx.on{opacity:.5;pointer-events:auto}
      #qx.on:hover{opacity:1;color:var(--accent)}
      /* Palace health line under the title: drawer and fact counts at a
         glance, so stagnation is visible without opening anything. */
      #hs{position:fixed;left:34px;top:72px;z-index:6;letter-spacing:.14em;
        text-transform:uppercase;opacity:.48}
      /* Knowledge-graph facts listed above search results. */
      #facts{margin-bottom:4px}
      .fact{padding:6px 0;border-bottom:1px solid rgba(var(--line),.16);
        font:400 10px/1.6 ui-monospace,Menlo,monospace;text-transform:none;
        letter-spacing:.02em;word-break:break-word}
      .fact b{color:var(--accent);font-weight:400}
      .fact.expired{opacity:.4;text-decoration:line-through}
      .fact small{display:block;opacity:.5;letter-spacing:.08em;
        text-transform:uppercase;margin-top:2px;text-decoration:none}
      #zc button.on{color:var(--accent)}
      /* Full-text popup: a sheet laid over the drawing, same paper and
         hairline border as the reading panel so it belongs to the same set. */
      #ov{position:fixed;inset:0;z-index:9;background:rgba(0,0,0,.38);display:none}
      #ov.on{display:block}
      #mo{position:fixed;z-index:10;left:50%;top:50%;
        transform:translate(-50%,-50%);width:min(760px,90vw);max-height:82vh;
        display:none;flex-direction:column;background:rgb(var(--knock));
        border:1px solid rgba(var(--line),.5);padding:26px 28px 22px}
      #mo.on{display:flex}
      #mo h3{font:400 11px/1.5 inherit;letter-spacing:.3em;
        text-transform:uppercase;margin-bottom:3px;padding-right:26px}
      #mo .meta{opacity:.55;letter-spacing:.16em;text-transform:uppercase;
        margin-bottom:14px;word-break:break-word}
      #mo .rule{height:1px;background:rgba(var(--line),.28);margin:0 0 14px}
      #mb{flex:1;overflow-y:auto;white-space:pre-wrap;word-break:break-word;
        text-transform:none;letter-spacing:.01em;
        font:400 11px/1.85 ui-monospace,Menlo,monospace}
      #mx{position:absolute;right:14px;top:12px;font:inherit;background:none;
        border:none;color:var(--ink);opacity:.5;cursor:pointer;padding:4px}
      #mx:hover{opacity:1;color:var(--accent)}
      #mf{margin-top:16px;padding-top:14px;
        border-top:1px solid rgba(var(--line),.28)}
      #mg{font:inherit;letter-spacing:.14em;text-transform:uppercase;
        background:rgba(var(--knock),.94);border:1px solid rgba(var(--line),.36);
        color:var(--ink);padding:5px 10px;cursor:pointer}
      #mg:hover{border-color:var(--accent);color:var(--accent)}
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

    const searchClear = document.createElement("button");
    searchClear.id = "qx";
    searchClear.type = "button";
    searchClear.textContent = "✕";
    searchClear.title = "Clear search";
    document.body.appendChild(searchClear);

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
    const diaryBtn = document.createElement("button");
    diaryBtn.type = "button";
    diaryBtn.id = "dy";
    diaryBtn.textContent = "Diary";
    zoomControls.append(zoomIn, zoomOut, fitBtn, themeBtn, diaryBtn);
    document.body.appendChild(zoomControls);

    const health = document.createElement("div");
    health.id = "hs";
    health.className = "c";
    document.body.appendChild(health);

    const banners = document.createElement("div");
    banners.id = "banner-slot";
    document.body.appendChild(banners);

    const panel = document.createElement("div");
    panel.id = "p";
    // Static skeleton only -- no interpolated data.
    panel.innerHTML =
      '<span id="x">✕</span>' +
      '<h3 id="ph"></h3>' +
      '<div class="meta" id="pm"></div>' +
      '<div id="facts"></div>' +
      '<div class="rule"></div>' +
      '<div class="body" id="pb"></div>' +
      '<div class="src" id="ps"></div>' +
      '<div class="rule"></div>' +
      '<div id="neighbours"></div>';
    document.body.appendChild(panel);

    const overlay = document.createElement("div");
    overlay.id = "ov";
    document.body.appendChild(overlay);

    const modal = document.createElement("div");
    modal.id = "mo";
    // Static skeleton only -- no interpolated data.
    modal.innerHTML =
      '<button type="button" id="mx">✕</button>' +
      '<h3 id="mh"></h3>' +
      '<div class="meta" id="mm"></div>' +
      '<div class="rule"></div>' +
      '<div id="mb"></div>' +
      '<div id="mf"><button type="button" id="mg">Show on map</button></div>';
    document.body.appendChild(modal);
  }

  buildUI();

  const canvas = $("c");
  const renderer = new window.Renderer(canvas);

  const state = {
    meta: null,
    selected: null,
    // Drawer id -> position in meta.drawers, so a server hit (which knows ids)
    // can be resolved into the index space the renderer and Knn share.
    indexById: new Map(),
  };

  /* ---- boot -------------------------------------------------------- */

  async function boot() {
    state.meta = await (await fetch("/api/index")).json();
    const buffer = await (await fetch("/api/vectors")).arrayBuffer();
    window.Knn.load(buffer, state.meta.drawer_count, state.meta.vector_dim);
    state.indexById = new Map(state.meta.drawers.map((d, i) => [d.id, i]));
    // Indexes built before the KG snapshot existed simply have no facts.
    state.kg = state.meta.kg || { entity_count: 0, triples: [] };
    renderer.setData(state.meta);

    const expired = state.kg.triples.filter((t) => t.to).length;
    $("hs").textContent =
      `${state.meta.drawer_count} drawers · ${state.kg.entity_count} entities · ` +
      `${state.kg.triples.length} facts` + (expired ? ` (${expired} expired)` : "");
    initTheme();
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
    // A star of relevance lines from the clicked drawer to each neighbour —
    // same colour/dotting as search, but rays from a centre rather than a path.
    renderer.chain = [];
    renderer.rays = neighbours.map((n) => ({ from: index, to: n.index, rel: 1 - n.distance }));

    $("ph").textContent = `${record.wing} / ${record.hall} / ${record.room}`;
    $("pm").textContent = record.date || "undated";
    $("facts").replaceChildren(); // facts belong to search mode only
    $("pb").textContent = record.text || "(empty drawer)";
    $("ps").textContent = record.source_file ? `source · ${record.source_file}` : "";
    $("p").classList.add("on");
    // New content, so start at the top of it -- otherwise the panel keeps
    // whatever scroll position the previous drawer left behind.
    $("p").scrollTop = 0;

    renderNeighbours(neighbours);
    renderer.draw();
  }

  /* Reading a result must not cost you the result list. Clicking a card opens
     the drawer's full text over the map; selecting it -- which rebuilds the
     panel, redraws the star and loses the results -- stays behind an explicit
     "Show on map". Neighbour cards keep selecting directly: wandering between
     related drawers is the point of that list. */
  let fullTextIndex = null;

  async function openFullText(index) {
    const drawer = drawerAt(index);
    if (!drawer) return;
    fullTextIndex = index;

    $("mh").textContent = `${drawer.wing} / ${drawer.hall} / ${drawer.room}`;
    $("mm").textContent = drawer.date || "undated";
    $("mb").textContent = "Loading…";
    $("ov").classList.add("on");
    $("mo").classList.add("on");
    $("mb").scrollTop = 0;
    $("mx").focus();

    const record = await (await fetch(`/api/drawer/${drawer.id}`)).json();
    // A second click while this was in flight owns the popup now.
    if (fullTextIndex !== index) return;
    $("mb").textContent = record.text || "(empty drawer)";
    if (record.source_file) {
      $("mm").textContent = `${drawer.date || "undated"} · ${record.source_file}`;
    }
  }

  function closeFullText() {
    fullTextIndex = null;
    $("ov").classList.remove("on");
    $("mo").classList.remove("on");
  }

  function closePanel() {
    state.selected = null;
    renderer.selected = null;
    renderer.highlighted = new Set();
    // The neighbour star belongs to the selection that drew it, so it has to
    // go with it -- otherwise the rays outlive the drawer they radiate from.
    renderer.rays = [];
    $("p").classList.remove("on");
    renderer.draw();
  }

  /* One card renderer behind both lists -- neighbours and search results -- so
     a card reads and behaves identically wherever it appears.

     Cards go up immediately with meta.json's preview so the list is never
     blank, then each body is swapped for a fuller snippet once the server has
     read it. The preview is only the first 200 characters, which for a log or
     a directory listing says nothing about what the drawer actually holds. */
  async function renderCards(rows, query) {
    const list = $("neighbours");
    list.replaceChildren();

    const bodies = new Map();
    rows.forEach((row) => {
      const d = drawerAt(row.index);
      if (!d) return;

      const card = document.createElement("div");
      card.className = "neighbour";

      const body = document.createElement("span");
      body.className = "text";
      body.textContent = d.preview || "";
      card.appendChild(body);
      bodies.set(d.id, body);

      const small = document.createElement("small");
      small.textContent = `${d.wing}/${d.room} · ${row.note}`;
      card.appendChild(small);

      card.addEventListener("click", () => openFullText(row.index));
      list.appendChild(card);
    });

    if (!bodies.size) return;

    const response = await fetch("/api/snippets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [...bodies.keys()], query }),
    });
    const { snippets } = await response.json();
    Object.entries(snippets || {}).forEach(([id, text]) => {
      const body = bodies.get(id);
      if (body) body.textContent = text;
    });
  }

  function renderNeighbours(neighbours) {
    // No query to centre on here, so a snippet is simply the head of the text.
    renderCards(
      neighbours.map((n) => ({ index: n.index, note: n.distance.toFixed(3) })),
      ""
    );
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

    // A search takes over the map from the diary lens.
    state.diaryOn = false;
    $("dy").classList.remove("on");

    const post = (url) =>
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed }),
      }).then((r) => r.json());

    // The two halves answer different questions. The embedding finds drawers
    // that mean something similar; the substring scan finds a literal token
    // the embedding has no reason to rank at all. An identifier like a ticket
    // number is invisible to the first and obvious to the second.
    const [semanticBody, literalBody] = await Promise.all([
      post("/api/search"),
      post("/api/search-text"),
    ]);

    const scores = window.Knn.similarities(Float32Array.from(semanticBody.vector));
    const drawers = state.meta.drawers;
    // Rank every drawn drawer by relevance (cosine similarity), strongest first.
    const ranked = drawers
      .map((_, i) => ({ index: i, rel: scores[i] }))
      .sort((a, b) => b.rel - a.rel);

    const literal = new Set(
      (literalBody.ids || []).map((id) => state.indexById.get(id)).filter((i) => i !== undefined)
    );
    // A name match is literal too: a room called "technical" answers a search
    // for "technical" even when no drawer body happens to say the word.
    const needle = trimmed.toLowerCase();
    drawers.forEach((d, i) => {
      if (
        d.wing.toLowerCase().includes(needle) ||
        d.hall.toLowerCase().includes(needle) ||
        d.room.toLowerCase().includes(needle)
      ) {
        literal.add(i);
      }
    });

    const semantic = ranked
      .filter((r) => r.rel >= RECALL_FLOOR && !literal.has(r.index))
      .slice(0, MATCH_MAX);
    const matches = new Set([...literal, ...semantic.map((r) => r.index)]);

    // The recall verdict: mempalace_search is pure semantic top-N, so the
    // first RECALL_TOP of the ranking -- literal hits included, floor applied
    // -- is exactly the set an agent would be handed for this query.
    const recall = ranked.slice(0, RECALL_TOP);
    const recalledCount = recall.filter((r) => r.rel >= RECALL_FLOOR).length;
    const recallRank = new Map(recall.map((r, k) => [r.index, k + 1]));

    // The relevance chain: a path through the strongest genuine hits, in rank
    // order. Weak noise is excluded by the floor; the renderer dots-out any
    // remaining segment below its own threshold.
    renderer.chain = ranked.filter((r) => r.rel >= CHAIN_FLOOR).slice(0, CHAIN_MAX);
    renderer.rays = []; // search draws the chain, not the neighbour star
    renderer.highlighted = matches;
    renderer.dimmed = new Set(drawers.map((_, i) => i).filter((i) => !matches.has(i)));
    $("qn").textContent = `${matches.size} of ${drawers.length} drawers`;
    $("qn").style.opacity = 1;

    renderer.draw();
    renderSearchResults(trimmed, literal, semantic, literalBody.truncated, {
      recalledCount,
      recallRank,
    });
  }

  /* Knowledge-graph facts whose subject or object mentions the query. This
     is the second recall leg (kg_query) made visible: what the agent KNOWS
     as structured fact, as opposed to what merely sits in a drawer. Expired
     facts render struck through -- invalidated knowledge is a health signal,
     not clutter to hide. */
  const FACTS_MAX = 12;

  function renderFacts(query) {
    const list = $("facts");
    list.replaceChildren();
    const needle = query.toLowerCase();
    const hits = state.kg.triples
      .filter(
        (t) =>
          t.s.toLowerCase().includes(needle) ||
          t.o.toLowerCase().includes(needle) ||
          t.p.toLowerCase().includes(needle)
      )
      .slice(0, FACTS_MAX);

    hits.forEach((t) => {
      const row = document.createElement("div");
      row.className = t.to ? "fact expired" : "fact";

      const subject = document.createElement("span");
      subject.textContent = t.s;
      const predicate = document.createElement("b");
      predicate.textContent = ` ${t.p.replaceAll("_", " ")} `;
      const object = document.createElement("span");
      object.textContent = t.o;
      row.append(subject, predicate, object);

      const when = document.createElement("small");
      when.textContent = t.to
        ? `fact · ${t.from || "?"} — expired ${t.to}`
        : `fact · since ${t.from || "?"}`;
      row.appendChild(when);
      list.appendChild(row);
    });
    return hits.length;
  }

  /* Search reuses the reading panel rather than adding a second list. The
     panel therefore has two modes -- one drawer (opened by select) or a result
     set (opened here) -- and both fill #neighbours with the same card markup,
     so a result behaves exactly like a neighbour: click it and you land in the
     drawer, which switches the panel back to its first mode. */
  function renderSearchResults(query, literal, semantic, truncated, verdict) {
    const rows = [
      ...[...literal].map((index) => ({ index, exact: true })),
      ...semantic.map((r) => ({ index: r.index, exact: false, rel: r.rel })),
    ].filter((row) => drawerAt(row.index) !== undefined);

    $("ph").textContent = `${rows.length} result${rows.length === 1 ? "" : "s"} · ${query}`;
    // The health line this panel exists for: would an agent asking MemPalace
    // this question get anything back it is allowed to use?
    const parts = [
      `claude recall: ${verdict.recalledCount} of top ${RECALL_TOP} ≥ ${RECALL_FLOOR}`,
    ];
    if (truncated) parts.push(`literal capped at ${literal.size}`);
    $("pm").textContent = parts.join(" · ");
    const factCount = renderFacts(query);
    $("pb").textContent = rows.length || factCount
      ? ""
      : "No drawer contains this text, and nothing clears the recall floor in meaning.";
    $("ps").textContent = "";

    $("p").classList.add("on");
    $("p").scrollTop = 0;
    renderCards(
      rows.map((row) => {
        const rank = verdict.recallRank.get(row.index);
        const base = row.exact ? "exact" : row.rel.toFixed(3);
        // A "claude #n" marker means: this exact drawer is one of the five
        // mempalace_search would hand over for this query.
        return { index: row.index, note: rank ? `${base} · claude #${rank}` : base };
      }),
      query
    );
  }

  /* ---- diary lens -------------------------------------------------------
     The third recall leg: the agent's own session journal. Diary entries are
     ordinary drawers in one wing, so the lens is a filter, not a data source
     -- dim everything else and list the entries newest-first. */
  const DIARY_WING = "wing_diary";

  function toggleDiary() {
    if (state.diaryOn) {
      state.diaryOn = false;
      $("dy").classList.remove("on");
      clearSearch();
      return;
    }

    state.diaryOn = true;
    $("dy").classList.add("on");
    const drawers = state.meta.drawers;
    const entries = drawers
      .map((d, i) => ({ index: i, date: d.date || "" }))
      .filter((e) => drawers[e.index].wing === DIARY_WING)
      .sort((a, b) => (a.date < b.date ? 1 : -1));

    const diarySet = new Set(entries.map((e) => e.index));
    renderer.highlighted = diarySet;
    renderer.dimmed = new Set(drawers.map((_, i) => i).filter((i) => !diarySet.has(i)));
    renderer.chain = [];
    renderer.rays = [];

    $("ph").textContent = `diary · ${entries.length} entries`;
    $("pm").textContent = entries.length
      ? `${entries[entries.length - 1].date.slice(0, 10)} → ${entries[0].date.slice(0, 10)}`
      : "";
    $("facts").replaceChildren();
    $("pb").textContent = entries.length ? "" : "No diary entries in this index.";
    $("ps").textContent = "";
    $("p").classList.add("on");
    $("p").scrollTop = 0;
    renderCards(
      entries.map((e) => ({ index: e.index, note: e.date.slice(0, 10) || "undated" })),
      ""
    );
    renderer.draw();
  }

  // The clear button only exists while there is something to clear.
  function syncClearButton() {
    $("qx").classList.toggle("on", $("q").value.length > 0);
  }

  function clearSearch() {
    renderer.dimmed = new Set();
    renderer.chain = [];
    $("q").value = "";
    syncClearButton();
    $("qn").style.opacity = 0;
    // closePanel does the rest: highlighted, rays, the open panel and the draw.
    closePanel();
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
    // A click on empty floor is a deselect: hitTest is the only signal for
    // "outside" on a canvas, since painted drawers can't bubble an event.
    if (hit === null) {
      closePanel();
      return;
    }
    select(hit);
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

  $("dy").addEventListener("click", toggleDiary);

  $("q").addEventListener("keydown", (event) => {
    if (event.key === "Enter") search(event.target.value);
    if (event.key === "Escape") clearSearch();
  });

  $("q").addEventListener("input", syncClearButton);

  $("qx").addEventListener("click", () => {
    clearSearch();
    $("q").focus();
  });

  $("x").addEventListener("click", closePanel);

  $("mx").addEventListener("click", closeFullText);
  $("ov").addEventListener("click", closeFullText);
  $("mg").addEventListener("click", () => {
    const index = fullTextIndex;
    closeFullText();
    if (index !== null) select(index);
  });

  // Escape closes the popup first; the search box's own Escape (which clears
  // the query) only applies once nothing is layered over the map.
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (fullTextIndex === null) return;
    event.stopPropagation();
    closeFullText();
  });

  window.__locium = { renderer, state, select, search, setTheme };
  boot();
})();
