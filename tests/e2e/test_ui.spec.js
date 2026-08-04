/* End-to-end coverage against the floorplan viewer, driven entirely through
   window.__locium and the renderer's state -- the canvas has no queryable
   DOM, so every assertion below reads state rather than pixels.

   Fixture shape (see fixture_server.py's LAYOUT): 6 wings, 9 (wing, hall)
   pairs, 10 (wing, hall, room) chambers, dot_cap=12 with two chambers over
   that cap so the capped path is exercised, plus a hall_diary/diary chamber
   inside a project wing for the diary lens -- the shape MemPalace actually
   writes -- and a two-fact knowledge graph beside the palace. */
const { test, expect } = require("@playwright/test");

const WING_COUNT = 6;
const HALL_COUNT = 9;
const CHAMBER_COUNT = 10;

async function ready(page) {
  await page.goto("/");
  await page.waitForFunction(() => window.__locium && window.__locium.state.meta);
}

test("the building renders with the fixture's wings, halls and chambers", async ({ page }) => {
  await ready(page);
  const meta = await page.evaluate(() => window.__locium.state.meta);

  expect(meta.wings.length).toBe(WING_COUNT);
  expect(meta.halls.length).toBe(HALL_COUNT);
  expect(meta.chambers.length).toBe(CHAMBER_COUNT);
  // Every block occupied means every wing/hall/chamber got a real rect.
  for (const rect of [...meta.wings, ...meta.halls, ...meta.chambers].map((g) => g.rect)) {
    expect(rect[2]).toBeGreaterThan(0);
    expect(rect[3]).toBeGreaterThan(0);
  }
  // The fixture deliberately puts two chambers over dot_cap.
  expect(meta.chambers.some((c) => c.capped)).toBe(true);

  // And over cluster_min, so saturated chambers carry labelled zones.
  const clustered = meta.chambers.filter((c) => c.clusters);
  expect(clustered.length).toBeGreaterThan(0);
  for (const c of clustered) {
    expect(c.clusters.length).toBeGreaterThanOrEqual(2);
    expect(c.clusters.reduce((s, z) => s + z.count, 0)).toBeGreaterThan(0);
  }
});

test("clicking a dot opens the reading panel with the full text", async ({ page }) => {
  await ready(page);
  const preview = await page.evaluate(() => window.__locium.state.meta.drawers[0].preview);
  await page.evaluate(() => window.__locium.select(0));

  const panelOn = await page.evaluate(() => document.getElementById("p").classList.contains("on"));
  expect(panelOn).toBe(true);

  const text = await page.locator("#pb").textContent();
  expect(text.length).toBeGreaterThan(preview.length);
});

test("selecting a drawer highlights ten neighbours, each a real drawer", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));

  const result = await page.evaluate(() => {
    const highlighted = [...window.__locium.renderer.highlighted];
    const drawers = window.__locium.state.meta.drawers;
    return {
      size: highlighted.length,
      allResolve: highlighted.every((i) => drawers[i] !== undefined),
    };
  });

  // The selected drawer plus its 10 nearest neighbours.
  expect(result.size).toBe(11);
  // The guard for the index-alignment bug: every highlighted index must
  // resolve to a real meta.drawers entry.
  expect(result.allResolve).toBe(true);
});

test("a neighbour card opens the same full-text popup as a search result", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));
  await page.waitForFunction(
    () => document.querySelectorAll("#neighbours .neighbour").length > 0
  );
  const before = await page.evaluate(() => window.__locium.state.selected);

  await page.locator("#neighbours .neighbour").first().click();
  await expect(page.locator("#mo")).toHaveClass(/on/);
  await page.waitForFunction(
    () => !document.getElementById("mb").textContent.startsWith("Loading")
  );

  // Reading a neighbour leaves the selection where it was; wandering to it is
  // the explicit "Show on map".
  expect(await page.evaluate(() => window.__locium.state.selected)).toBe(before);
  await page.click("#mg");
  await page.waitForFunction((b) => window.__locium.state.selected !== b, before);
});

test("hovering a result marks its dot without moving the camera", async ({ page }) => {
  await ready(page);
  await searchWithResults(page, "wing_a");
  const before = await page.evaluate(() => ({
    scale: window.__locium.renderer.scale,
    x: window.__locium.renderer.offsetX,
    y: window.__locium.renderer.offsetY,
  }));

  await page.locator("#neighbours .neighbour").first().hover();
  expect(await page.evaluate(() => window.__locium.renderer.hovered)).not.toBeNull();

  // The marker appears; the viewport must NOT chase the cursor.
  const during = await page.evaluate(() => ({
    scale: window.__locium.renderer.scale,
    x: window.__locium.renderer.offsetX,
    y: window.__locium.renderer.offsetY,
  }));
  expect(during).toEqual(before);

  await page.locator("#ph").hover();
  expect(await page.evaluate(() => window.__locium.renderer.hovered)).toBeNull();
});

test("clicking a result opens the popup and flies the map to its dot", async ({ page }) => {
  await ready(page);
  await searchWithResults(page, "wing_a");

  const scaleBefore = await page.evaluate(() => window.__locium.renderer.scale);
  await page.locator("#neighbours .neighbour").first().click();
  await expect(page.locator("#mo")).toHaveClass(/on/);

  // The flight settles centred on the clicked drawer's dot, marker alive
  // under the popup -- at the user's zoom level, which the flight never touches.
  await page.waitForFunction(() => {
    const r = window.__locium.renderer;
    if (r.hovered === null) return false;
    const d = window.__locium.state.meta.drawers[r.hovered];
    const [sx, sy] = r.worldToScreen(d.x, d.y);
    return Math.abs(sx - r.viewW / 2) < 1 && Math.abs(sy - r.viewH / 2) < 1;
  });
  expect(await page.evaluate(() => window.__locium.renderer.scale)).toBe(scaleBefore);

  // Closing the popup releases the marker.
  await page.keyboard.press("Escape");
  expect(await page.evaluate(() => window.__locium.renderer.hovered)).toBeNull();
});

test("the panel scrolls back to the top when its content is replaced", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));
  await page.waitForFunction(
    () => document.querySelectorAll("#neighbours .neighbour").length > 0
  );

  // Scroll down, then swap the content: the new drawer must start at the top.
  await page.evaluate(() => {
    document.getElementById("p").scrollTop = 400;
  });
  expect(await page.evaluate(() => document.getElementById("p").scrollTop)).toBeGreaterThan(0);

  const neighbour = await page.evaluate(() =>
    [...window.__locium.renderer.highlighted].find((i) => i !== window.__locium.state.selected)
  );
  await page.evaluate((i) => window.__locium.select(i), neighbour);
  expect(await page.evaluate(() => document.getElementById("p").scrollTop)).toBe(0);
});

test("selecting a drawer draws a neighbour star, cleared by search", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));

  const star = await page.evaluate(() => {
    const rays = window.__locium.renderer.rays;
    return {
      count: rays.length,
      allFromCentre: rays.every((r) => r.from === 0),
      relsInRange: rays.every((r) => r.rel >= 0 && r.rel <= 1),
      chainEmpty: window.__locium.renderer.chain.length === 0,
    };
  });
  // A ray from the clicked drawer to each of its ten neighbours.
  expect(star.count).toBe(10);
  expect(star.allFromCentre).toBe(true);
  expect(star.relsInRange).toBe(true);
  // The star and the search chain are mutually exclusive views.
  expect(star.chainEmpty).toBe(true);

  await page.evaluate(() => window.__locium.search("technical"));
  expect(await page.evaluate(() => window.__locium.renderer.rays.length)).toBe(0);
});

/* Deselect is the one path that cannot be driven through window.__locium: it
   lives in the canvas pointerup handler, so these tests issue a real click.
   "Empty floor" means a point that hit-tests to null AND is not covered by a
   panel -- pointerdown has to land on the canvas for the handler to arm. */
async function emptyFloorPoint(page) {
  return page.evaluate(() => {
    const canvas = document.getElementById("c");
    const rect = canvas.getBoundingClientRect();
    const renderer = window.__locium.renderer;
    for (let y = 12; y < rect.height - 12; y += 7) {
      for (let x = 12; x < rect.width - 12; x += 7) {
        if (renderer.hitTest(x, y) !== null) continue;
        if (document.elementFromPoint(rect.left + x, rect.top + y) !== canvas) continue;
        return { x: rect.left + x, y: rect.top + y };
      }
    }
    return null;
  });
}

test("clicking empty floor clears the selection and its neighbour star", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));
  expect(await page.evaluate(() => window.__locium.renderer.rays.length)).toBe(10);

  const point = await emptyFloorPoint(page);
  expect(point).not.toBeNull();
  await page.mouse.click(point.x, point.y);

  const cleared = await page.evaluate(() => ({
    selected: window.__locium.state.selected,
    rendererSelected: window.__locium.renderer.selected,
    rays: window.__locium.renderer.rays.length,
    highlighted: window.__locium.renderer.highlighted.size,
    panelOn: document.getElementById("p").classList.contains("on"),
  }));

  // Nothing may survive the deselect -- least of all the rays, which are what
  // the selection actually draws on the floorplan.
  expect(cleared.selected).toBeNull();
  expect(cleared.rendererSelected).toBeNull();
  expect(cleared.rays).toBe(0);
  expect(cleared.highlighted).toBe(0);
  expect(cleared.panelOn).toBe(false);
});

test("closing the panel with the ✕ also clears the neighbour star", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));
  expect(await page.evaluate(() => window.__locium.renderer.rays.length)).toBe(10);

  await page.click("#x");

  expect(await page.evaluate(() => window.__locium.renderer.rays.length)).toBe(0);
  expect(await page.evaluate(() => window.__locium.state.selected)).toBeNull();
});

async function searchWithResults(page, query) {
  await page.evaluate((q) => window.__locium.search(q), query);
  await page.waitForFunction(
    () => document.querySelectorAll("#neighbours .neighbour").length > 0
  );
}

test("search lists its hits in the panel", async ({ page }) => {
  await ready(page);
  await searchWithResults(page, "wing_a");

  expect(await page.evaluate(() => document.getElementById("p").classList.contains("on"))).toBe(
    true
  );
  expect(await page.locator("#ph").textContent()).toContain("wing_a");
});

test("a result opens its full text without disturbing the selection", async ({ page }) => {
  await ready(page);
  await searchWithResults(page, "wing_a");

  await page.locator("#neighbours .neighbour").first().click();
  await expect(page.locator("#mo")).toHaveClass(/on/);
  await page.waitForFunction(
    () => !document.getElementById("mb").textContent.startsWith("Loading")
  );
  expect((await page.locator("#mb").textContent()).length).toBeGreaterThan(0);

  // Reading a result costs neither the selection nor the result list.
  expect(await page.evaluate(() => window.__locium.state.selected)).toBeNull();
  expect(await page.locator("#neighbours .neighbour").count()).toBeGreaterThan(0);

  // Selecting stays an explicit opt-in.
  await page.click("#mg");
  await expect(page.locator("#mo")).not.toHaveClass(/on/);
  await page.waitForFunction(() => window.__locium.state.selected !== null);
  expect(await page.evaluate(() => window.__locium.renderer.rays.length)).toBe(10);
});

test("chunks of one exchange collapse to a single result card", async ({ page }) => {
  await ready(page);
  // "pipeline" literally occurs in split0 AND split2 -- two chunks of the
  // same exchange, and nowhere else in the fixture.
  await page.evaluate(() => window.__locium.search("pipeline"));
  await page.waitForFunction(
    () => document.querySelectorAll("#neighbours .neighbour").length > 0
  );

  const cards = await page.evaluate(() =>
    [...document.querySelectorAll("#neighbours .neighbour small")].map((s) => s.textContent)
  );
  expect(cards.length).toBe(1);
  expect(cards[0]).toContain("3-part exchange");
  // The map still highlights every matching chunk -- only the list dedupes.
  expect(await page.evaluate(() => window.__locium.renderer.highlighted.size)).toBe(2);
});

test("the verdict reports the recall gap against whole messages", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.search("record sent pipeline"));
  await page.waitForFunction(() => document.getElementById("p").classList.contains("on"));

  const verdict = await page.locator("#pm").textContent();
  expect(verdict).toMatch(/claude recall: \d of top 5 ≥ 0.15/);
  expect(verdict).toMatch(/whole messages would give \d/);
});

test("noisy results collapse behind one line, expandable on demand", async ({ page }) => {
  await ready(page);
  // "noisetoken" occurs only in the tool-dump slices: all three are one
  // document family, and their content classifies as tool output.
  await page.evaluate(() => window.__locium.search("noisetoken"));
  await page.waitForFunction(() => document.getElementById("p").classList.contains("on"));

  expect(await page.locator("#neighbours .neighbour").count()).toBe(0);
  const toggle = page.locator("#more-results");
  await expect(toggle).toHaveText(/\+ 1 data \/ tool-output result/);

  await toggle.click();
  const note = await page.locator("#neighbours .neighbour small").first().textContent();
  expect(note).toContain("3-part exchange");
  expect(note).toContain("tool output");
  expect(await page.locator("#more-results").count()).toBe(0);
});

test("the panel previews a long drawer; full text lives in the popup", async ({ page }) => {
  await ready(page);
  // The stitched tool-dump message (~1k chars) exceeds the panel preview
  // limit, so selecting one of its slices must show a trimmed body plus the
  // read-full affordance instead of the whole text.
  await page.evaluate(() => {
    const index = window.__locium.state.indexById.get("dump1");
    window.__locium.select(index);
  });
  await page.waitForFunction(() =>
    document.getElementById("pb").classList.contains("open")
  );

  const preview = await page.evaluate(() => document.getElementById("pb").textContent);
  expect(preview.length).toBeLessThan(800);

  // The trimmed body itself is the click target, like a result card.
  await page.click("#pb");
  await expect(page.locator("#mo")).toHaveClass(/on/);
  await page.waitForFunction(
    () => !document.getElementById("mb").textContent.startsWith("Loading")
  );
  const full = await page.evaluate(() => document.getElementById("mb").textContent);
  expect(full.length).toBeGreaterThan(preview.length);
  expect(await page.locator("#mm").textContent()).toContain("stitched from 3 drawers");

  // The slice that led here (dump1, part 2 of 3) is marked inside the
  // stitched message -- and it is the MIDDLE occurrence, not a text match:
  // all three dump slices carry identical text, only the offset tells them
  // apart.
  const mark = await page.evaluate(() => {
    const m = document.querySelector("#mb mark");
    return m
      ? { text: m.textContent, before: m.previousSibling.textContent.length }
      : null;
  });
  expect(mark).not.toBeNull();
  expect(mark.text).toContain("noisetoken");
  expect(mark.before).toBe(mark.text.length + 1); // one identical slice + "\n"
});

test("a split exchange reads stitched back together", async ({ page }) => {
  await ready(page);
  // Select the MIDDLE chunk: on its own it starts mid-word ("nt.php ...").
  await page.evaluate(() => {
    const index = window.__locium.state.indexById.get("split1");
    window.__locium.select(index);
  });
  await page.waitForFunction(
    () => !document.getElementById("pb").textContent.includes("(empty drawer)")
      && document.getElementById("pb").textContent.length > 0
  );

  const body = await page.locator("#pb").textContent();
  const meta = await page.locator("#pm").textContent();
  // Concatenation heals the word the miner split.
  expect(body).toContain("RecordSentEvent.php");
  expect(body).toContain("closes the exchange");
  expect(meta).toContain("stitched from 3 drawers");
});

test("the full-text popup closes on escape and on the backdrop", async ({ page }) => {
  await ready(page);
  await searchWithResults(page, "wing_a");

  await page.locator("#neighbours .neighbour").first().click();
  await expect(page.locator("#mo")).toHaveClass(/on/);
  await page.keyboard.press("Escape");
  await expect(page.locator("#mo")).not.toHaveClass(/on/);

  await page.locator("#neighbours .neighbour").first().click();
  await expect(page.locator("#mo")).toHaveClass(/on/);
  await page.locator("#ov").click({ position: { x: 5, y: 5 } });
  await expect(page.locator("#mo")).not.toHaveClass(/on/);
});

test("a query nothing answers reports an honest recall verdict", async ({ page }) => {
  await ready(page);
  // No drawer contains this literally, and the fixture's drawer vectors are
  // random, so at most a stray dot clears the recall floor. Before the floor
  // existed this lit the top 40 by rank regardless. The map, the list and the
  // headline must all tell the same story.
  await page.evaluate(() => window.__locium.search("zzqx-no-such-token-90210"));
  await page.waitForFunction(() => document.getElementById("p").classList.contains("on"));

  const state = await page.evaluate(() => ({
    highlighted: window.__locium.renderer.highlighted.size,
    cards: document.querySelectorAll("#neighbours .neighbour").length,
    header: document.getElementById("ph").textContent,
    verdict: document.getElementById("pm").textContent,
    exacts: [...document.querySelectorAll("#neighbours .neighbour small")].filter((s) =>
      s.textContent.includes("exact")
    ).length,
  }));

  expect(state.cards).toBe(state.highlighted);
  expect(state.header).toContain(`${state.cards} result`);
  expect(state.exacts).toBe(0);
  expect(state.verdict).toMatch(/claude recall: \d of top 5/);
});

test("the search box clear button resets the query and the map", async ({ page }) => {
  await ready(page);
  // The button is only offered when there is something to clear.
  await expect(page.locator("#qx")).not.toHaveClass(/on/);

  await page.fill("#q", "wing_a");
  await page.dispatchEvent("#q", "input");
  await expect(page.locator("#qx")).toHaveClass(/on/);

  await page.press("#q", "Enter");
  await page.waitForFunction(() => window.__locium.renderer.dimmed.size > 0);

  await page.click("#qx");
  const cleared = await page.evaluate(() => ({
    value: document.getElementById("q").value,
    dimmed: window.__locium.renderer.dimmed.size,
    highlighted: window.__locium.renderer.highlighted.size,
    rays: window.__locium.renderer.rays.length,
    panelOn: document.getElementById("p").classList.contains("on"),
  }));

  expect(cleared.value).toBe("");
  expect(cleared.dimmed).toBe(0);
  expect(cleared.highlighted).toBe(0);
  expect(cleared.rays).toBe(0);
  expect(cleared.panelOn).toBe(false);
});

test("the health line reports drawers, entities and facts", async ({ page }) => {
  await ready(page);
  const line = await page.locator("#hs").textContent();
  // Tool traffic reports under its own label, not as an umbrella "noise"
  // share that silently included data payloads too.
  expect(line).toMatch(/\d+ drawers · 3 entities · 2 facts \(1 expired\) · \d+% noise/);
});

test("a search that names an entity lists its facts, expired ones struck", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.search("wing_a"));
  await page.waitForFunction(() => document.querySelectorAll("#facts .fact").length > 0);

  const facts = await page.evaluate(() =>
    [...document.querySelectorAll("#facts .fact")].map((f) => ({
      text: f.textContent,
      expired: f.classList.contains("expired"),
    }))
  );
  expect(facts.length).toBe(2);
  expect(facts.find((f) => f.text.includes("depends on")).expired).toBe(false);
  expect(facts.find((f) => f.text.includes("legacy-runner")).expired).toBe(true);
});

test("the diary lens filters the map to diary entries, newest first", async ({ page }) => {
  await ready(page);
  await page.click("#dy");
  await page.waitForFunction(
    () => document.querySelectorAll("#neighbours .neighbour").length > 0
  );

  const state = await page.evaluate(() => ({
    header: document.getElementById("ph").textContent,
    highlighted: window.__locium.renderer.highlighted.size,
    dimmed: window.__locium.renderer.dimmed.size,
    total: window.__locium.state.meta.drawers.length,
    notes: [...document.querySelectorAll("#neighbours .neighbour small")].map(
      (s) => s.textContent
    ),
  }));
  expect(state.header).toContain("diary · 4 entries");
  expect(state.highlighted).toBe(4);
  expect(state.dimmed).toBe(state.total - 4);
  // Newest first: the date notes must be non-increasing.
  const dates = state.notes.map((n) => n.trim());
  expect([...dates].sort().reverse()).toEqual(dates);

  // Toggling off restores the full map.
  await page.click("#dy");
  expect(await page.evaluate(() => window.__locium.renderer.dimmed.size)).toBe(0);
});

test("search dims non-matches without moving any dot", async ({ page }) => {
  await ready(page);
  const before = await page.evaluate(() =>
    window.__locium.state.meta.drawers.map((d) => [d.x, d.y])
  );

  await page.evaluate(() => window.__locium.search("wing_a"));
  await page.waitForFunction(() => window.__locium.renderer.dimmed.size > 0);

  const after = await page.evaluate(() =>
    window.__locium.state.meta.drawers.map((d) => [d.x, d.y])
  );
  expect(after).toEqual(before);

  const highlightedSize = await page.evaluate(() => window.__locium.renderer.highlighted.size);
  expect(highlightedSize).toBeGreaterThan(0);
});

test("search builds a relevance chain, ranked and cleared correctly", async ({ page }) => {
  await ready(page);
  // The fixture plants four graded semantic hits for this query; the rest of
  // its vectors are random and never clear the chain floor. Searching for
  // anything else leaves the chain empty and the assertions below vacuous.
  await page.evaluate(() => window.__locium.search("deployment rollback checklist"));
  await page.waitForFunction(() => window.__locium.renderer.highlighted.size > 0);

  const chain = await page.evaluate(() =>
    window.__locium.renderer.chain.map((c) => ({ index: c.index, rel: c.rel }))
  );
  // Every hit clears the floor, the chain is capped, and it is ordered by
  // relevance strongest-first (this is what the connecting lines thread through).
  expect(chain.length).toBeGreaterThanOrEqual(4);
  expect(chain.length).toBeLessThanOrEqual(16);
  for (const c of chain) {
    expect(c.rel).toBeGreaterThanOrEqual(0.3);
    expect(c.index).toBeGreaterThanOrEqual(0);
    expect(c.index).toBeLessThan(await page.evaluate(() => window.__locium.state.meta.drawers.length));
  }
  for (let i = 1; i < chain.length; i += 1) {
    expect(chain[i - 1].rel).toBeGreaterThanOrEqual(chain[i].rel);
  }

  // Selecting a drawer is the wander, not a search — the chain must clear.
  await page.evaluate(() => window.__locium.select(0));
  expect(await page.evaluate(() => window.__locium.renderer.chain.length)).toBe(0);

  // And a fresh search then a clear leaves no chain behind.
  await page.evaluate(() => window.__locium.search("technical"));
  await page.evaluate(() => window.__locium.search(""));
  expect(await page.evaluate(() => window.__locium.renderer.chain.length)).toBe(0);
});

test("clearing search restores every drawer", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.search("wing_a"));
  await page.waitForFunction(() => window.__locium.renderer.dimmed.size > 0);

  await page.evaluate(() => window.__locium.search(""));

  const { dimmed, highlighted } = await page.evaluate(() => ({
    dimmed: window.__locium.renderer.dimmed.size,
    highlighted: window.__locium.renderer.highlighted.size,
  }));
  expect(dimmed).toBe(0);
  expect(highlighted).toBe(0);
});

test("zoom changes the renderer's scale", async ({ page }) => {
  await ready(page);
  const before = await page.evaluate(() => window.__locium.renderer.scale);

  await page.evaluate(() => {
    const r = window.__locium.renderer;
    r.zoomBy(2, r.viewW / 2, r.viewH / 2);
  });

  const after = await page.evaluate(() => window.__locium.renderer.scale);
  expect(after).toBeGreaterThan(before);
});

test("the theme toggle switches and persists across a reload", async ({ page }) => {
  await ready(page);
  const initial = await page.evaluate(() => window.__locium.renderer.themeName);
  const target = initial === "dark" ? "light" : "dark";

  await page.evaluate((t) => window.__locium.setTheme(t), target);
  expect(await page.evaluate(() => window.__locium.renderer.themeName)).toBe(target);

  await page.reload();
  await page.waitForFunction(() => window.__locium && window.__locium.state.meta);
  expect(await page.evaluate(() => window.__locium.renderer.themeName)).toBe(target);
});

test("wandering to a neighbour moves the selection", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));
  expect(await page.evaluate(() => window.__locium.state.selected)).toBe(0);

  const neighbour = await page.evaluate(() =>
    [...window.__locium.renderer.highlighted].find((i) => i !== window.__locium.state.selected)
  );
  await page.evaluate((i) => window.__locium.select(i), neighbour);
  expect(await page.evaluate(() => window.__locium.state.selected)).toBe(neighbour);
});
