import { test, expect } from "@playwright/test";

test("search, empty state and resetting filters", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Follow your curiosity." }),
  ).toBeVisible();
  const firstTitle = await page.locator(".book-card h3").first().innerText();
  const search = page.getByRole("searchbox");
  await search.fill(firstTitle);
  await search.press("Enter");
  await expect(page.locator(".book-card h3").first()).toHaveText(firstTitle);
  await search.fill("zzzxxyythisbookdoesnotexist");
  await search.press("Enter");
  await expect(
    page.getByRole("heading", { name: "No books on this path. Yet." }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Explore all books", exact: true })
    .click();
  await expect(page.locator(".book-card")).toHaveCount(16);
  const genre = page.locator(".genre-option").nth(1);
  await genre.click();
  await expect(genre).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".active-chip")).toHaveCount(1);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.locator(".active-chip")).toHaveCount(0);
});

test("saved books and favourites persist after reload", async ({ page }) => {
  await page.goto("/");
  const firstCard = page.locator(".book-card").first();
  const title = await firstCard.locator("h3").innerText();
  await firstCard
    .getByRole("button", { name: `Save ${title}`, exact: true })
    .click();
  await page.getByRole("button", { name: /My shelf/ }).click();
  await expect(page.locator(".book-card h3")).toHaveText(title);
  await page.reload();
  await page.getByRole("button", { name: /My shelf/ }).click();
  await expect(page.locator(".book-card h3")).toHaveText(title);
  await page
    .getByRole("button", { name: `Add ${title} to favourites`, exact: true })
    .click();
  await page.getByRole("button", { name: "Favourites 1", exact: true }).click();
  await expect(page.locator(".book-card h3")).toHaveText(title);
  await page
    .getByRole("button", { name: `View ${title}`, exact: true })
    .click();
  await page
    .getByRole("button", { name: "Hide this book from discovery", exact: true })
    .click();
  await page.getByRole("button", { name: "Hidden 1", exact: true }).click();
  await expect(page.locator(".book-card h3")).toHaveText(title);
  await page.getByRole("button", { name: "Restore", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "No books hidden away." }),
  ).toBeVisible();
});

test("book details restore keyboard focus and seed recommendations", async ({
  page,
}) => {
  await page.goto("/");
  const firstCard = page.locator(".book-card").first();
  const title = await firstCard.locator("h3").innerText();
  const opener = firstCard.getByRole("button", {
    name: `View ${title}`,
    exact: true,
  });
  await opener.click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("dialog").getByRole("heading", { name: title, exact: true }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(opener).toBeFocused();
  await opener.click();
  await page
    .getByRole("button", { name: "Close book details", exact: true })
    .click();
  await expect(opener).toBeFocused();
  await firstCard
    .getByRole("button", { name: `More like ${title}`, exact: true })
    .click();
  await expect(page.locator(".active-chip")).toContainText(
    `More like ${title}`,
  );
  await expect(
    page.locator(".book-card h3").filter({
      hasText: new RegExp(`^${title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`),
    }),
  ).toHaveCount(0);
});

test("invalid saved state is harmless and layout fits the viewport", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("leximind.reading-shelf.v1", "not-json"),
  );
  await page.goto("/");
  await expect(page.locator(".book-card")).toHaveCount(16);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: /My shelf/ }).click();
  await expect(
    page.getByRole("heading", { name: "A shelf full of possibilities." }),
  ).toBeVisible();
  await expect(
    page.getByText("Your shelf stays in this browser.", { exact: false }),
  ).toBeVisible();
});

test("shareable book page has sources, related books and a real missing-page response", async ({
  page,
}) => {
  await page.goto("/");
  const title = await page.locator(".book-card h3").first().innerText();
  await page.locator(".book-card .cover-button").first().click();
  const source = await page
    .getByRole("dialog")
    .getByRole("link", { name: "View on Open Library" })
    .getAttribute("href");
  await page.getByRole("link", { name: "Open book page", exact: true }).click();
  await expect(page).toHaveURL(/\/books\/OL\d+W$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(title);
  await expect(page).toHaveTitle(`${title} — LexiMind`);
  await expect(
    page.getByRole("link", { name: "View on Open Library" }),
  ).toHaveAttribute("href", source!);
  await expect(
    page.getByRole("heading", { name: "A few more doors to open." }),
  ).toBeVisible();
  expect(
    await page.locator(".related-books .book-card").count(),
  ).toBeGreaterThan(0);
  await page
    .getByRole("link", { name: "Open in discovery", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("dialog").getByRole("heading", { name: title, exact: true }),
  ).toBeVisible();
  const missing = await page.goto("/books/OL0W");
  expect(missing?.status()).toBe(404);
  await expect(
    page.getByRole("heading", { name: "This chapter is missing." }),
  ).toBeVisible();
});
