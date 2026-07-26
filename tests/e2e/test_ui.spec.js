/* End-to-end coverage against the floorplan viewer, driven entirely through
   window.__locium and the renderer's state -- the canvas has no queryable
   DOM, so every assertion below reads state rather than pixels.

   Fixture shape (see fixture_server.py's LAYOUT): 6 wings, 8 (wing, hall)
   pairs, 9 (wing, hall, room) chambers, dot_cap=12 with two chambers over
   that cap so the capped path is exercised. */
const { test, expect } = require("@playwright/test");

const WING_COUNT = 6;
const HALL_COUNT = 8;
const CHAMBER_COUNT = 9;

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
  await page.evaluate(() => window.__locium.search("technical architecture notes"));
  await page.waitForFunction(() => window.__locium.renderer.highlighted.size > 0);

  const chain = await page.evaluate(() =>
    window.__locium.renderer.chain.map((c) => ({ index: c.index, rel: c.rel }))
  );
  // Every hit clears the floor, the chain is capped, and it is ordered by
  // relevance strongest-first (this is what the connecting lines thread through).
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

test("the trail grows as you wander and does not accumulate duplicates", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => window.__locium.select(0));
  expect(await page.evaluate(() => window.__locium.state.trail.length)).toBe(1);

  const neighbour = await page.evaluate(() =>
    [...window.__locium.renderer.highlighted].find((i) => i !== window.__locium.state.selected)
  );
  await page.evaluate((i) => window.__locium.select(i), neighbour);
  expect(await page.evaluate(() => window.__locium.state.trail.length)).toBe(2);

  // Re-selecting the same drawer must not grow the trail.
  await page.evaluate((i) => window.__locium.select(i), neighbour);
  expect(await page.evaluate(() => window.__locium.state.trail.length)).toBe(2);
});
