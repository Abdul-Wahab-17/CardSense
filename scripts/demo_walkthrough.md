# CardSense Blackjack — viva demo walkthrough

A 5-minute demo script for the distributed real-time Blackjack assistant. Stick to it
unless someone interrupts with a question.

## Pre-flight checklist (do 30 min before)

- [ ] Docker Desktop running (whale icon in tray, "Engine running")
- [ ] Stack up: `WORKERS=2 ./scripts/start.sh` — wait for "Backend is healthy"
- [ ] Smoke test passes: `./scripts/smoke_blackjack.sh`
- [ ] OpenRouter key set in `.env` (`LLM_API_KEY=sk-or-...`)
- [ ] Real deck of cards on the desk
- [ ] Phone propped overhead with a stand or on a tall stack of books, pointing
      down at the play area. Connect to same Wi-Fi as laptop. Bookmark the
      laptop's LAN IP + port 8000.
- [ ] Decent table-top lighting — a desk lamp pointed at the play area is
      worth more than 50 lines of code.
- [ ] Pre-recorded `demo_video.mp4` saved next to the repo as a fallback
      input for the scaling demo (see step 4 below).

---

## Step 1 — context (30s)

> "We built a distributed real-time card-game assistant. Imagine the brain
> behind a pair of AR glasses that watch the cards on the table, count them,
> and tell you what to do. Today we're showing it on a phone-camera-as-glasses,
> playing Blackjack."

Show the dashboard at `http://<laptop-ip>:8000/blackjack.html`.

## Step 2 — start a game + draw ROIs (45s)

1. Click **Start Game**.
2. Click **Start Camera** → grant permission.
3. Click **Start Stream**. (frames begin uploading)
4. Click-drag once to mark the **dealer region** (top of frame).
5. Click-drag again to mark the **player region** (bottom of frame).

Narrate while drawing:

> "I'm telling the system where the dealer's cards will appear and where mine
> will. The system uses these regions to classify each detected card as
> player or dealer in real time."

## Step 3 — play one hand live (90s)

Lay out:
- Two cards in dealer region (one face up, one face down — represented by
  any face-down card; system will only see the upcard until you flip).
- Two cards in your region.

The HUD on the right should populate:
- Both your cards listed
- Dealer's upcard listed; hole still shown as "?"
- **Big "HIT" or "STAND" or "DOUBLE" recommendation** with the reason

Narrate:

> "The system already knows what's in front of me. It's seen [my cards]
> versus the dealer's [upcard]. It's recommending [action] because [reason
> from the HUD]. The running count is [N], which is [low|neutral|high]."

Take the recommended action physically. Place the next card. The HUD
updates within a second or two.

If you bust, push, win — state the outcome and click **Tipsi review** to
get the LLM's post-hand commentary (~3 seconds, then displays).

## Step 4 — distributed scaling demo (90s)

Open a terminal on the laptop:

```bash
docker compose up -d --scale worker-service=1
```

Open the URL `http://<laptop-ip>:8000/api/workers/status` — show 1 worker
heartbeating.

Run the edge client to pump frames from a recorded video:

```bash
python scripts/edge_camera_client.py \
  --base-url http://localhost:8000 \
  --camera-id load-1 \
  --source ./demo_video.mp4 \
  --fps 15
```

Watch the queue build up. Then scale:

```bash
docker compose up -d --scale worker-service=4
```

Refresh `/api/workers/status` — show 4 workers. Watch the queue drain.

> "The system distributes camera frames across multiple worker nodes. Adding
> workers increases throughput. Each game's state is consistent because we
> use a per-game lock and idempotent state transitions."

## Step 5 — architecture pitch (30s)

Optional slide or whiteboard sketch:

```
Browser/Glasses → FastAPI backend → Redis Streams → Worker pool (YOLO)
                                          ↑              ↓
                                    state lock    detections
                                          ↓              ↓
                                  Blackjack Engine (state, count, recommendation)
                                          ↓
                        WebSocket → Browser HUD (live)   LLM (Tipsi narration)
```

Talking points:
- **Concurrency:** multiple frames in flight across workers
- **Scalability:** horizontal worker scaling via docker-compose
- **Consistency:** per-game lock, idempotent transitions, position-based dedup
  of detections so a card sitting on the table for 30 frames doesn't generate
  30 observations
- **Smart features built on top:** Hi-Lo card counting, Illustrious-18
  deviations, basic strategy lookup, LLM-narrated explanations + post-hand
  coaching

## Step 6 — Q&A buffer (60s)

Likely questions:

**Q: What card game does this work for?**
A: Pluggable. We've shipped Blackjack today; the same pipeline runs Thulla
(a 4-player trick-taking game with opponent modeling) — same workers, same
queue, different game-specific recommender.

**Q: How accurate is detection?**
A: We use a pretrained YOLOv8 playing-card model. Accuracy is good in
typical lighting on standard decks. Cards must be face-up and not heavily
occluded. We filter and dedup detections in two stages: per-frame IoU
clustering and cross-frame position-based tracking.

**Q: What if a worker crashes?**
A: Redis Streams has consumer groups; surviving workers continue from the
group's position. (If they ask about message recovery: in-flight messages
of a crashed worker can be reclaimed via XAUTOCLAIM — that's a known
follow-up enhancement.)

**Q: Why one camera instead of two?**
A: Real card games are played on a single surface. The system uses ROI
rectangles to classify each detection as player or dealer based on its
bounding-box position. This is also what a glasses-mounted camera would do.

**Q: Can it count cards?**
A: Yes. The Hi-Lo running and true count are both visible in the top bar.
The recommendation switches between basic strategy and the
Illustrious 18 count-based deviations automatically.

**Q: Where does the LLM fit in?**
A: Two places. First, an optional narration of the recommendation in
plain English. Second, "Tipsi" — a post-hand teach mode that reviews what
you did and points out anything you could've played differently. Both
fall back gracefully to the deterministic recommender if the LLM is
unavailable.
