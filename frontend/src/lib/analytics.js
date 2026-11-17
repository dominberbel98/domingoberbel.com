/**
 * Visitor analytics with Umami Cloud.
 *
 * Umami sets no cookies and stores nothing in the browser: it tells visitors
 * apart with a hash of IP and user agent that rotates, so there is no consent
 * banner and every visit is counted, not only the ones that click "accept".
 *
 * Three things keep the numbers honest:
 *   - With no VITE_UMAMI_WEBSITE_ID at build time this whole module is a no-op.
 *   - `data-domains` makes the tracker send only from the production host, so
 *     local builds and previews never count.
 *   - Opening the site once with `?notrack=1` excludes that browser from now on
 *     (`?notrack=0` undoes it). That keeps the site's owner out of the counts on
 *     a phone, where there is no console to set the flag by hand.
 */

const WEBSITE_ID = (import.meta.env.VITE_UMAMI_WEBSITE_ID || "").trim();
const SCRIPT_URL = "https://cloud.umami.is/script.js";
const PRODUCTION_HOST = "www.domingoberbel.com";
// The tracker itself checks this key before sending anything.
const OPT_OUT_KEY = "umami.disabled";

let started = false;
// Sections shown before the tracker finished loading; sent once it has.
const pending = [];

export function analyticsConfigured() {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(WEBSITE_ID);
}

function applyOptOutFromUrl() {
  try {
    const flag = new URLSearchParams(window.location.search).get("notrack");
    if (flag === "1") window.localStorage.setItem(OPT_OUT_KEY, "1");
    if (flag === "0") window.localStorage.removeItem(OPT_OUT_KEY);
  } catch {
    /* private browsing or a malformed query string */
  }
}

/** Call once on start-up. */
export function initAnalytics() {
  if (started || !analyticsConfigured()) return;
  started = true;
  applyOptOutFromUrl();

  const script = document.createElement("script");
  script.defer = true;
  script.src = SCRIPT_URL;
  script.dataset.websiteId = WEBSITE_ID;
  script.dataset.domains = PRODUCTION_HOST;
  script.addEventListener("load", () => {
    while (pending.length) trackSection(pending.shift());
  });
  document.head.appendChild(script);
}

/**
 * The site is one page whose sections swap in place, so Umami's automatic
 * page view counts the visit but not what was looked at. This reports each
 * section shown as a `section_view` event.
 */
export function trackSection(section) {
  if (!started) return;
  if (window.umami?.track) window.umami.track("section_view", { section });
  else pending.push(section);
}
