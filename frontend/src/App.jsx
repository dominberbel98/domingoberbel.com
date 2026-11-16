import React, { Suspense, lazy, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Chat from "./components/Chat";
import { motionAllowed } from "./components/Motion";
import ThumbsUp from "./components/ThumbsUp";
import { useLanguage, useT } from "./i18n";
import { getTelemetry, loadBackendMeta, subscribeTelemetry } from "./lib/telemetry";

// Only the chat is in the first download. Every other section — and the
// charting library three of them pull in — loads when it is first opened.
const Visualizaciones = lazy(() => import("./components/Visualizaciones"));
const ModelosPredictivos = lazy(() => import("./components/ModelosPredictivos"));
const ModelosScoring = lazy(() => import("./components/ModelosScoring"));
const ModelosRecomendacion = lazy(() => import("./components/ModelosRecomendacion"));
const Certificaciones = lazy(() => import("./components/Certificaciones"));
const Futboard = lazy(() => import("./components/futboard/Futboard"));

const MODEL_IDS = ["prediccion_la_liga", "modelo_scoring", "modelo_recomendation"];

/**
 * The navigation, built during render.
 *
 * This was a module-level `NAV_ITEMS` constant, which froze every label at
 * import time. Harmless with one language; a silent bug with two — switching to
 * Spanish would have translated the whole site except the menu.
 */
function useNavItems(tr) {
  return useMemo(
    () => [
      { id: "chat_cv", icon: "chat", label: tr.nav.chat_cv },
      { id: "visualizaciones", icon: "monitoring", label: tr.nav.visualizaciones },
      {
        id: "modelos",
        icon: "functions",
        label: tr.nav.modelos,
        children: [
          { id: "prediccion_la_liga", icon: "sports_soccer", label: tr.nav.prediccion_la_liga },
          { id: "modelo_scoring", icon: "credit_score", label: tr.nav.modelo_scoring },
          { id: "modelo_recomendation", icon: "recommend", label: tr.nav.modelo_recomendation },
        ],
      },
      { id: "certificaciones", icon: "workspace_premium", label: tr.nav.certificaciones },
      { id: "futboard", icon: "sports_soccer", label: tr.nav.futboard },
    ],
    [tr],
  );
}

function NavList({ items, activeSection, modelsOpen, onToggleModels, onSelect, hover, indicator = false }) {
  const baseRow = "flex items-center gap-3 px-6 py-4 cursor-pointer active:scale-95";
  const subRow  = "flex items-center gap-3 pl-12 pr-6 py-3 cursor-pointer active:scale-95 text-[0.86rem]";
  const transition = hover ? " transition-colors" : "";
  const activeCls   = "bg-[#00FF41]/10 text-[#00FF41] border-l-4 border-[#00FF41]";
  const inactiveCls = (sub) =>
    `${sub ? "text-[#00FF41]/60" : "text-[#00FF41]/65"} ${
      hover ? "hover:bg-[#00FF41]/5 hover:text-[#00FF41]" : ""
    } border-l-4 border-transparent`;
  // With the sliding indicator, the row it rests on keeps the active text
  // colour but leaves the highlight to the indicator, or there would be two.
  const targetCls = "text-[#00FF41] border-l-4 border-transparent";
  const currentCls = indicator ? targetCls : activeCls;
  const target = indicator ? { "data-nav-target": "" } : {};

  return items.map((item) => {
    if (item.children) {
      const isActive = MODEL_IDS.includes(activeSection);
      // Collapsed with a model open: the group row is where the indicator
      // rests. Expanded, it rests on the child and the group stays lit.
      const groupIsTarget = isActive && !modelsOpen;
      return (
        <div key={item.id}>
          <div
            onClick={onToggleModels}
            {...(groupIsTarget ? target : {})}
            className={`${baseRow}${transition} ${
              isActive ? (groupIsTarget ? currentCls : activeCls) : inactiveCls(false)
            }`}
          >
            <span className="material-symbols-outlined text-[1.2rem]">{item.icon}</span>
            <span className="flex-1">{item.label}</span>
            <span className={`material-symbols-outlined text-[1rem] transition-transform ${modelsOpen ? "rotate-180" : ""}`}>
              expand_more
            </span>
          </div>
          {modelsOpen && item.children.map((c) => (
            <div
              key={c.id}
              onClick={() => onSelect(c.id)}
              {...(activeSection === c.id ? target : {})}
              className={`${subRow}${transition} ${activeSection === c.id ? currentCls : inactiveCls(true)}`}
            >
              <span className="material-symbols-outlined text-[1rem]">{c.icon}</span>
              <span>{c.label}</span>
            </div>
          ))}
        </div>
      );
    }
    return (
      <div
        key={item.id}
        onClick={() => onSelect(item.id)}
        {...(activeSection === item.id ? target : {})}
        className={`${baseRow}${transition} ${activeSection === item.id ? currentCls : inactiveCls(false)}`}
      >
        <span className="material-symbols-outlined text-[1.2rem]">{item.icon}</span>
        <span>{item.label}</span>
      </div>
    );
  });
}

/**
 * The site-wide language switch.
 *
 * Both codes are always shown rather than a single "switch to X" button, so it
 * reads as a state you can see rather than an action whose result you have to
 * guess. Sized for a thumb, because on a phone it sits in the header next to the
 * hamburger.
 */
function LanguageSwitch({ language, onChange, label, compact = false }) {
  return (
    <div
      role="group"
      aria-label={label}
      className="shrink-0 flex border border-[#00FF41]/40 rounded overflow-hidden"
    >
      {["en", "es"].map((code) => (
        <button
          key={code}
          type="button"
          onClick={() => onChange(code)}
          aria-pressed={language === code}
          className={`font-headline uppercase tracking-widest transition-colors ${
            compact ? "min-h-[44px] min-w-[44px] px-2.5 text-[0.82rem]" : "min-h-[44px] min-w-[44px] px-3 text-[0.88rem]"
          } ${
            language === code
              ? "bg-[#00FF41]/20 text-[#00FF41] font-bold drop-shadow-[0_0_6px_rgba(0,255,65,0.4)]"
              : "text-[#00FF41]/60 hover:text-[#00FF41] hover:bg-[#00FF41]/5"
          }`}
        >
          {code}
        </button>
      ))}
    </div>
  );
}

/** The explainer body, shared by the sidebar dock and the mobile sheet. */
function MascotBody({ tr, large = false }) {
  return (
    <>
      <p className={`text-[#00FF41]/75 leading-relaxed normal-case ${large ? "text-[0.92rem]" : "text-[0.84rem]"}`}>
        {tr.app.mascotLead}
      </p>
      <dl className={large ? "mt-4 space-y-3" : "mt-2.5 space-y-1.5"}>
        {tr.app.mascotSections.map((section) => (
          <div key={section.label}>
            <dt className={`text-[#00FF41]/90 font-bold tracking-wide ${large ? "text-[0.95rem]" : "text-[0.8rem]"}`}>
              {section.label}
            </dt>
            <dd className={`text-[#00FF41]/65 leading-relaxed normal-case ${large ? "text-[0.95rem]" : "text-[0.8rem]"}`}>
              {section.text}
            </dd>
          </div>
        ))}
      </dl>
    </>
  );
}

const API_URL =
  import.meta.env.VITE_API_URL ||
  `${window.location.protocol}//api.${window.location.hostname.replace(/^www\./, "")}`;

const MASCOT_SEEN_KEY = "site.mascotSeen";

export default function App() {
  const tr = useT();
  const { language, setLanguage } = useLanguage();
  const navItems = useNavItems(tr);

  const [menuOpen, setMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState("chat_cv");
  const [modelsOpen, setModelsOpen] = useState(false);
  const [telemetry, setTelemetryState] = useState(getTelemetry);
  const [mascotVisible, setMascotVisible] = useState(true);

  // ── Motion ────────────────────────────────────────────────────────────────
  const mainRef = useRef(null);
  const navRef = useRef(null);
  const slideTimers = useRef([]);
  const lastIndicated = useRef("chat_cv");
  // "" | "leave" | "enter": drives the slide between sections (styles.css).
  const [sectionPhase, setSectionPhase] = useState("");
  const [navBox, setNavBox] = useState(null);
  // Each increment plays the thumbs-up once; 0 means it is not on screen.
  const [thumbsRun, setThumbsRun] = useState(0);

  // The avatar pops out of the footer bar shortly after the page loads, and
  // again whenever the assistant finishes an answer.
  useEffect(() => {
    if (!motionAllowed()) return undefined;
    const id = setTimeout(() => setThumbsRun((n) => n + 1), 1400);
    return () => clearTimeout(id);
  }, []);
  useEffect(() => {
    if (telemetry.lastLatencyMs != null && motionAllowed()) setThumbsRun((n) => n + 1);
  }, [telemetry.lastLatencyMs]);

  /**
   * Change section. With motion allowed the current one slides out first and
   * the new one slides in; otherwise it is the plain cut it always was. A
   * second click mid-slide cancels the first and goes straight to the new one.
   */
  const selectSection = (id) => {
    slideTimers.current.forEach(clearTimeout);
    slideTimers.current = [];
    if (id === activeSection) {
      setSectionPhase("");
      return;
    }
    if (!motionAllowed()) {
      setActiveSection(id);
      return;
    }
    setSectionPhase("leave");
    slideTimers.current = [
      setTimeout(() => {
        setActiveSection(id);
        setSectionPhase("enter");
      }, 150),
      setTimeout(() => setSectionPhase(""), 150 + 280),
    ];
  };

  useEffect(() => () => slideTimers.current.forEach(clearTimeout), []);

  // The sidebar highlight glides to the active entry. It only animates when
  // the section changed: when the models group merely opens or closes, the
  // rows jump, so the highlight jumps with them.
  useLayoutEffect(() => {
    const measure = (animate) => {
      const row = navRef.current?.querySelector("[data-nav-target]");
      setNavBox(row ? { top: row.offsetTop, height: row.offsetHeight, animate } : null);
    };
    measure(lastIndicated.current !== activeSection);
    lastIndicated.current = activeSection;
    const remeasure = () => measure(false);
    window.addEventListener("resize", remeasure);
    document.fonts?.ready.then(remeasure).catch(() => {});
    return () => window.removeEventListener("resize", remeasure);
  }, [activeSection, modelsOpen, language]);

  // The dot grid lights up around the pointer, and the top edge of the panel
  // under it follows along. One listener, one write per frame.
  useEffect(() => {
    if (!motionAllowed() || !window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
      return undefined;
    }
    let frame = 0;
    let x = 0;
    let y = 0;
    let panel = null;
    const onMove = (event) => {
      x = event.clientX;
      y = event.clientY;
      panel = event.target instanceof Element ? event.target.closest(".viz-panel") : null;
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const main = mainRef.current;
        if (main) {
          const box = main.getBoundingClientRect();
          main.style.setProperty("--mx", `${x - box.left}px`);
          main.style.setProperty("--my", `${y - box.top}px`);
        }
        if (panel) {
          panel.style.setProperty("--px", `${x - panel.getBoundingClientRect().left}px`);
        }
      });
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      cancelAnimationFrame(frame);
    };
  }, []);

  // On a phone there is no sidebar, so the explainer lives behind a header
  // button. It used to open by itself over the whole page on a first visit,
  // which meant the first thing a visitor saw was a wall of text with the chat
  // hidden behind it. Now the page is the first thing, and the button carries
  // a pulsing dot until the explainer has been opened once.
  const [mascotSheetOpen, setMascotSheetOpen] = useState(false);
  const [mascotSeen, setMascotSeen] = useState(() => {
    try {
      return Boolean(window.localStorage.getItem(MASCOT_SEEN_KEY));
    } catch {
      return false;
    }
  });

  const closeMascotSheet = () => {
    setMascotSheetOpen(false);
    setMascotSeen(true);
    try {
      window.localStorage.setItem(MASCOT_SEEN_KEY, "1");
    } catch {
      /* private browsing */
    }
  };

  // The footer reports what the backend actually is, so ask it once and then
  // follow the store that Chat updates after every answer.
  useEffect(() => {
    loadBackendMeta(API_URL);
    return subscribeTelemetry(setTelemetryState);
  }, []);

  useEffect(() => {
    if (MODEL_IDS.includes(activeSection)) setModelsOpen(true);
  }, [activeSection]);

  // A sheet that covers the page must not leave the page scrolling behind it.
  useEffect(() => {
    if (!mascotSheetOpen) return undefined;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [mascotSheetOpen]);

  return (
    <>
      {/* Scanline CRT overlay */}
      <div className="scanline-overlay" />

      {/* Matrix background */}
      <div className="matrix-bg">
        <div className="matrix-dots" />
        <div className="math-overlay">
{`∑(x-μ)²/N  y=mx+b  ∇f(x)=0  P(A|B)=P(B|A)P(A)/P(B)  ∂f/∂x  ∫e^x dx
λ=f(x,y)  σ=√Var(X)  β=(X'X)⁻¹X'y  z=(x-μ)/σ  R²=1-SSR/SST  log(p/1-p)
H₀: μ₁=μ₂  α=0.05  tanh(x)  ReLU(x)  softmax(zᵢ)=e^ᶻⁱ/∑e^ᶻʲ  E=mc²
∑(x-μ)²/N  y=mx+b  ∇f(x)=0  P(A|B)=P(B|A)P(A)/P(B)  ∂f/∂x  ∫e^x dx
λ=f(x,y)  σ=√Var(X)  β=(X'X)⁻¹X'y  z=(x-μ)/σ  R²=1-SSR/SST  log(p/1-p)`}
        </div>
      </div>

      {/* Top App Bar */}
      <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center gap-1.5 sm:gap-2 px-3 sm:px-6 h-14 sm:h-16 bg-[#0e0e0e]/80 backdrop-blur-xl border-b border-[#00FF41]/15 shadow-[0_0_15px_0_rgba(0,255,65,0.15)]">
        {/* On a phone the owner's name takes the brand's place: the name is on
            the right of the bar on desktop, and there is no room for both here. */}
        <div className="min-w-0 truncate text-[0.95rem] sm:text-xl font-bold text-[#00FF41] drop-shadow-[0_0_8px_rgba(0,255,65,0.4)] font-headline tracking-tight md:tracking-tighter">
          <span className="md:hidden">{tr.app.owner}</span>
          <span className="hidden md:inline">{tr.app.brand}</span>
        </div>

        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          {/* Explainer, mobile only — the desktop sidebar has it docked. */}
          <button
            type="button"
            onClick={() => setMascotSheetOpen(true)}
            aria-label={tr.app.mascotOpen}
            title={tr.app.mascotOpen}
            className="md:hidden relative w-11 h-11 flex items-center justify-center rounded border border-[#00FF41]/30 text-[#00FF41] hover:bg-[#00FF41]/10 active:scale-95"
          >
            <span className="material-symbols-outlined text-[1.3rem]">smart_toy</span>
            {!mascotSeen && (
              <span className="absolute -top-1 -right-1 flex h-3 w-3" aria-hidden="true">
                <span className="absolute inline-flex h-full w-full rounded-full bg-[#00FF41] opacity-70 motion-safe:animate-ping" />
                <span className="relative inline-flex h-3 w-3 rounded-full bg-[#00FF41]" />
              </span>
            )}
          </button>

          <LanguageSwitch
            language={language}
            onChange={setLanguage}
            label={tr.app.languageLabel}
            compact
          />

          {/* Hamburger — mobile only */}
          <button
            className="md:hidden w-11 h-11 flex flex-col items-center justify-center gap-[5px] pointer-events-auto"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label={tr.app.menuLabel}
          >
            <span className={`block w-5 h-[2px] bg-[#00FF41] transition-all duration-200 ${menuOpen ? "rotate-45 translate-y-[7px]" : ""}`} />
            <span className={`block w-5 h-[2px] bg-[#00FF41] transition-all duration-200 ${menuOpen ? "opacity-0" : ""}`} />
            <span className={`block w-5 h-[2px] bg-[#00FF41] transition-all duration-200 ${menuOpen ? "-rotate-45 -translate-y-[7px]" : ""}`} />
          </button>

          <nav className="hidden md:flex items-center">
            <div className="font-headline uppercase tracking-[0.25em] text-[0.95rem] text-[#00FF41] drop-shadow-[0_0_10px_rgba(0,255,65,0.6)] font-bold">
              {tr.app.owner}
            </div>
          </nav>
        </div>
      </header>

      {/* Mobile dropdown menu */}
      {menuOpen && (
        <div className="md:hidden fixed top-14 left-0 w-full z-[45] bg-[#0e0e0e]/95 backdrop-blur-xl border-b border-[#00FF41]/15 font-headline text-[0.86rem] uppercase max-h-[calc(100vh-56px)] overflow-y-auto scrollbar-hide">
          <div className="px-4 py-3 border-b border-[#00FF41]/10">
            <div className="text-[#00FF41] font-bold">{tr.app.workspace}</div>
            <div className="text-[#00FF41]/65 tracking-widest text-[0.78rem] mt-0.5">{tr.app.session}</div>
          </div>
          <NavList
            items={navItems}
            activeSection={activeSection}
            modelsOpen={modelsOpen}
            onToggleModels={() => setModelsOpen((o) => !o)}
            onSelect={(id) => { selectSection(id); setMenuOpen(false); }}
            hover={false}
          />
        </div>
      )}

      {/* Tap-outside overlay to close the menu */}
      {menuOpen && (
        <div
          className="md:hidden fixed inset-0 z-[44]"
          onClick={() => setMenuOpen(false)}
        />
      )}

      {/* Explainer sheet — mobile only, covers the page */}
      {mascotSheetOpen && (
        <div className="md:hidden fixed inset-0 z-[70] bg-[#050505]/95 backdrop-blur-sm flex flex-col">
          <div className="flex items-center gap-3 px-4 h-14 border-b border-[#00FF41]/20 shrink-0">
            <span className="material-symbols-outlined text-[#00FF41] text-2xl mascot-bob drop-shadow-[0_0_8px_rgba(0,255,65,0.6)] shrink-0">
              smart_toy
            </span>
            <span className="mascot-antenna h-1.5 w-1.5 bg-[#00FF41] rounded-full shrink-0" />
            <h2 className="flex-1 min-w-0 truncate text-[#00FF41] font-headline font-bold text-[1rem]">
              {tr.app.mascotTitle}
            </h2>
            {/* The same switch as the header, so the explanation can be read in
                either language without leaving the sheet. It sets the site
                language, so the change survives closing it. */}
            <LanguageSwitch
              language={language}
              onChange={setLanguage}
              label={tr.app.languageLabel}
              compact
            />
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-hide px-4 py-5">
            <MascotBody tr={tr} large />
          </div>

          <div className="p-4 border-t border-[#00FF41]/20 shrink-0">
            <button
              type="button"
              onClick={closeMascotSheet}
              className="w-full min-h-[52px] border border-[#00FF41]/60 rounded bg-[#00FF41]/15 hover:bg-[#00FF41]/25 active:scale-[0.98] font-headline font-bold uppercase tracking-widest text-[0.95rem] text-[#00FF41]"
            >
              {tr.app.mascotClose}
            </button>
          </div>
        </div>
      )}

      {/* Side Nav — desktop */}
      <aside className="fixed left-0 top-16 bottom-8 hidden md:flex flex-col z-40 bg-[#0e0e0e] w-72 border-r border-[#00FF41]/15 font-headline text-[0.9rem] uppercase overflow-y-auto scrollbar-hide">
        <div className="p-6 border-b border-[#00FF41]/10">
          <div className="text-[#00FF41] font-bold text-lg">{tr.app.workspace}</div>
          <div className="text-[#00FF41]/65 tracking-widest mt-1">{tr.app.session}</div>
        </div>
        <div ref={navRef} className="relative py-4 shrink-0">
          {navBox && (
            <span
              aria-hidden="true"
              className={`nav-indicator${navBox.animate ? " nav-indicator--glide" : ""}`}
              style={{ transform: `translateY(${navBox.top}px)`, height: navBox.height }}
            />
          )}
          <NavList
            items={navItems}
            activeSection={activeSection}
            modelsOpen={modelsOpen}
            onToggleModels={() => setModelsOpen((o) => !o)}
            onSelect={selectSection}
            hover
            indicator
          />
        </div>

        {/*
          Mascot, docked in the sidebar's own empty space.

          It was `fixed top-[72%] left-[20px] z-[60]` over a `z-40` sidebar that
          is `left-0 w-72`, so it drew on top of the navigation — worst on a short
          laptop screen, where 72% landed on the menu items. Moving it into the
          content area only traded one overlap for another: it then covered the
          chat composer. The sidebar has ample unused vertical space below four
          nav items, and nothing else competes for it, so it cannot overlap
          anything here at any viewport width.

          `mt-auto` parks it at the bottom when there is spare room, and the
          sidebar itself scrolls when there is not. Giving the nav its own
          scroll area instead made the two compete for a fixed height: with
          `models` expanded, the last menu entry was clipped behind the dock
          with no visible scrollbar to explain where it had gone.
        */}
        {mascotVisible && (
          <div className="mascot-dock mt-auto shrink-0 border-t border-[#00FF41]/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[#00FF41] text-2xl mascot-bob drop-shadow-[0_0_8px_rgba(0,255,65,0.6)] shrink-0">
                smart_toy
              </span>
              <span className="mascot-antenna h-1.5 w-1.5 bg-[#00FF41] rounded-full shrink-0" />
              <span className="text-[#00FF41] text-[0.88rem] font-bold tracking-wide normal-case">
                {tr.app.mascotTitle}
              </span>
              <button
                type="button"
                onClick={() => setMascotVisible(false)}
                aria-label={tr.app.mascotDismiss}
                title={tr.app.mascotDismiss}
                className="ml-auto w-5 h-5 flex items-center justify-center text-[#00FF41]/70 hover:text-[#00FF41] hover:bg-[#00FF41]/10 rounded shrink-0 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#00FF41]"
              >
                ✕
              </button>
            </div>

            <div className="mt-2">
              <MascotBody tr={tr} />
            </div>
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main
        ref={mainRef}
        className={`fixed left-0 md:left-72 top-14 sm:top-16 right-0 bottom-8 overflow-hidden flex flex-col items-center justify-center p-2 sm:p-6 md:p-8 bg-surface${
          sectionPhase ? ` section-${sectionPhase}` : ""
        }`}
      >
        {/* The page's one heading. The visible title of each section is an
            h2, and the name in the header is decoration inside a banner. */}
        <h1 className="sr-only">{tr.app.pageHeading}</h1>
        {activeSection === "chat_cv" && <Chat />}
        <Suspense
          fallback={
            <div role="status" className="font-headline uppercase tracking-widest text-[0.85rem] text-[#00FF41]/70 flicker">
              {tr.app.loadingSection}
            </div>
          }
        >
          {activeSection === "visualizaciones" && <Visualizaciones />}
          {activeSection === "prediccion_la_liga" && <ModelosPredictivos />}
          {activeSection === "modelo_scoring" && <ModelosScoring />}
          {activeSection === "modelo_recomendation" && <ModelosRecomendacion />}
          {activeSection === "certificaciones" && <Certificaciones />}
          {activeSection === "futboard" && <Futboard />}
        </Suspense>
        {/* Radial glow behind chat. `main-fx` layers stay still while the
            section itself slides. */}
        <div className="main-fx absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] z-[-2] pointer-events-none opacity-20">
          <div className="glow-drift w-full h-full bg-[radial-gradient(circle_at_center,_#00FF4133_0%,_transparent_70%)]" />
        </div>
        <div className="main-fx dots-lit" />
        <div className="main-fx crt-beam" />
      </main>

      {/* Thumbs-up: only beside the chat, where the corner is empty. */}
      {thumbsRun > 0 && activeSection === "chat_cv" && (
        <ThumbsUp key={thumbsRun} onDone={() => setThumbsRun(0)} />
      )}

      {/* Footer Stats */}
      <footer className="fixed bottom-0 left-0 w-full h-8 bg-surface-container border-t border-[#00FF41]/10 px-3 sm:px-6 flex items-center justify-between z-50 text-[0.72rem] sm:text-[0.74rem] font-headline uppercase text-[#00FF41]/65">
        <div className="flex gap-2 sm:gap-4">
          <span>
            {telemetry.documents == null
              ? tr.footer.indexUnknown
              : tr.footer.index(telemetry.documents)}
          </span>
          {telemetry.chatModel && (
            <span className="hidden md:inline">{tr.footer.model(telemetry.chatModel)}</span>
          )}
          {telemetry.embeddingModel && (
            <span className="hidden lg:inline">
              {tr.footer.embeddings(telemetry.embeddingModel)}
            </span>
          )}
        </div>
        <div className="flex gap-2 sm:gap-4 items-center">
          {telemetry.firstTokenMs != null && (
            <span key={telemetry.firstTokenMs} className="value-flash hidden lg:inline">
              {tr.footer.firstToken(telemetry.firstTokenMs)}
            </span>
          )}
          {telemetry.lastLatencyMs != null && (
            <span key={telemetry.lastLatencyMs} className="value-flash hidden sm:inline">
              {tr.footer.latency(telemetry.lastLatencyMs)}
            </span>
          )}
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-primary flicker" />
            {tr.footer.running}
          </span>
          <span className="hidden sm:inline">{tr.footer.domain}</span>
        </div>
      </footer>
    </>
  );
}
