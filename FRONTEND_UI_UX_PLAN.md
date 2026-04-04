# Frontend UI/UX Plan (Thulla CV Assistant)

## 1) Product intent
Build a clean, competitive-game assistant UI that helps one player make better legal moves in real time from two camera feeds:
- Hand camera: what is in your hand now.
- Table camera: current trick, played cards, turn flow.

The interface must prioritize decision speed, confidence clarity, and rule legality over developer diagnostics.

## 2) What the UI should do
- Show current game readiness in one glance: backend, realtime link, stream state, and freshness.
- Let user set game context quickly: seats, leader, to-play, lead suit, trump/rule toggles.
- Show current hand as clear card tiles with confidence and suit grouping.
- Show best action as a prominent primary recommendation wsudo reflector --country India,Singapore,UAE --latest 20 --protocol https --sort rate --save /etc/pacman.d/mirrorlistith legal rationale.
- Show 2-4 alternatives with EV/confidence deltas.
- Show concise trick/table summary: lead suit, cards on table, who is to play, won piles.
- Show short strategic guidance in plain language.
- Show recent activity timeline in human-readable format (not raw JSON dumps).

## 3) What the UI should NOT do
- Should not expose raw debug JSON by default.
- Should not show worker internals/log spam in the main gameplay view.
- Should not make calibration/config areas dominate primary gameplay surfaces.
- Should not force users to parse engineering terms to understand recommendation quality.
- Should not hide legal constraints (must-follow, must-trump-if-void) from decision context.

## 4) Visual system (target)

### 4.1 Color palette (table-first, competitive)
Use CSS variables:
- `--bg-0: #07140f` (page deep tone)
- `--bg-1: #0e2018` (secondary backdrop)
- `--panel: #12271f` (card background)
- `--panel-strong: #163429` (active panel)
- `--line: #2b4c3d` (borders)
- `--text: #eef6ef` (primary text)
- `--muted: #a7c2b0` (secondary text)
- `--accent: #d8ad58` (primary action, recommendation highlight)
- `--accent-2: #61c89b` (good state)
- `--warn: #f2c14e` (warning state)
- `--bad: #e57373` (error state)
- Suit accents:
  - `--suit-hearts: #f26d78`
  - `--suit-diamonds: #f2a86d`
  - `--suit-spades: #86b9ff`
  - `--suit-clubs: #7fd4a3`

### 4.2 Typography
- Display/headings: `"Space Grotesk", "Sora", "Segoe UI", sans-serif`
- Body/UI: `"IBM Plex Sans", "Segoe UI", sans-serif`
- Numerics/confidence chips: `"JetBrains Mono", "Consolas", monospace`

Scale:
- Page title: `32px / 700`
- Section title: `20px / 650`
- Card heading: `17px / 600`
- Body: `14-16px / 400-500`
- Meta labels: `12px / 500`

### 4.3 Spacing and shape
- Base spacing unit: `8px`
- Panel padding: `16-20px`
- Gap between major panels: `16px`
- Radius: `14px` panels, `999px` chips
- Border: `1px solid var(--line)`

## 5) Layout blueprint

### 5.1 Desktop (>=1200px)
Two-column workspace:
- Left rail (`32-36%`): controls and stream setup.
- Right stage (`64-68%`): recommendation hero, hand grid, table/trick, timeline.

Top sticky strip:
- Health chips only: backend, realtime, stream, last event age, model.

### 5.2 Tablet (768-1199px)
- Single-column stack with recommendation panel pinned first.
- Controls collapsible as “Game Controls”.

### 5.3 Mobile (<768px)
- Vertical flow.
- Sticky “Best Move” mini bar.
- Hand grid 2 columns, recommendation alternatives as compact rows.

## 6) Screen composition and order
1. Header: app name + one-line intent.
2. Status strip: system/game readiness.
3. Recommendation Hero:
   - Best card, confidence, EV, legality badge.
   - Why list (3 bullets max).
4. Your Hand:
   - 13 slots, grouped by suit, confidence per card.
5. Table/Trick State:
   - Lead suit, to-play, cards on table, won pile counts.
6. Strategy Notes:
   - High-level suggestions from engine.
7. Activity Timeline:
   - Recent events in readable lines.
8. Control Rail:
   - Camera registration/stream controls.
   - Game config controls.
   - Advanced section (collapsed by default): calibration/manual override.

## 7) Interaction patterns
- Primary action style only for start/confirm actions.
- Secondary neutral buttons for reset/clear.
- No more than one destructive action visible at once.
- Inline validation for suit inputs (`S/H/D/C`).
- Empty states with guidance (not technical errors).
- Polling freshness badge:
  - `Fresh <2.5s` green
  - `Stale 2.5-6s` yellow
  - `Old >6s` red

## 8) Recommendation presentation rules
- Always show legal scope first:
  - `Legal plays: N` and `Must follow suit: Yes/No`
- Best action block fields:
  - `best_card_label`
  - `best_action_ev`
  - `best_action_confidence`
  - `recommendation_mode`
  - `freshness_ts_ms`
- Alternatives list fields:
  - `card`
  - `ev`
  - `confidence`
  - `delta_vs_best`

Display logic:
- If confidence < threshold (e.g., 0.55), show “Low confidence” banner and soften recommendation language.
- If legal set uncertain, show “Conservative mode” label and avoid assertive wording.

## 9) Debug cleanup policy
Remove from default visible UI:
- Raw `debugInfo` JSON
- Full worker heartbeat panel
- Worker logs panel
- Full simulation JSON dumps
- Full diagnostics/belief JSON dumps

Allowed:
- A hidden `Advanced Diagnostics` accordion for developer use.
- Explicitly off by default and visually separated from gameplay area.

## 10) Accessibility and usability
- Color contrast >= WCAG AA for text and chips.
- Keyboard focus rings on all inputs/buttons.
- Hit targets >= 40px.
- Live region for recommendation changes.
- Do not rely on color only; include icons/text labels for statuses.

## 11) Frontend implementation checklist
- [ ] Define design tokens in `styles.css` (`:root` variables).
- [ ] Rebuild page into two-zone layout (`controls` + `game stage`).
- [ ] Create recommendation hero component section.
- [ ] Replace raw JSON areas with compact cards/lists.
- [ ] Group hand cards by suit and confidence.
- [ ] Convert status bar to concise health chips.
- [ ] Move calibration/manual override into collapsed Advanced panel.
- [ ] Keep only one activity timeline section for recent events.
- [ ] Add freshness badges and low-confidence banner logic.
- [ ] Add responsive breakpoints for desktop/tablet/mobile.
- [ ] Add consistent button hierarchy and disabled states.
- [ ] Add accessibility focus styles and semantic labels.

## 12) Definition of done (frontend)
Frontend is considered production-ready for gameplay support when:
- Core screens are decision-first and readable in <2 seconds.
- No debug/internal JSON is visible in default mode.
- Recommendation panel clearly communicates confidence and legality.
- Hand/table information is understandable without technical context.
- UI remains stable and usable at 720p laptop and mobile widths.
- All critical actions (register, stream, start/reset game, apply config) are reachable within 2 clicks.
