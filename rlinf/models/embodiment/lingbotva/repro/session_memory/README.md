# session_memory — raw working log (lab notebook)

Backup of the agent's persistent **auto-memory** for the LingBot-VA GRPO effort,
preserved here because the original lives in `~/.claude/` (deleted on container
teardown). This is the unedited, chronological "lab notebook" — every fix,
diagnosis, dead-end, and decision in the order it happened.

- **`lingbotva-rl-grpo.md`** — the full project memory: design decisions, the
  environment build, every bug + root cause + fix (FSDP double-shard, pidfd/IPC,
  mean-reduce log-prob, recompute-under-replay, the ragged-buffer deadlock), the
  tuning history, and the final exact-gradient result. Dense and not polished.
- **`MEMORY.md`** — the one-line index that pointed to it.

## How this relates to the curated docs
For reproduction and handoff, prefer the curated docs one level up — they distill
this log into something actionable:
- `../README.md` — index + TL;DR commands
- `../ENV_PREP.md` — environment build
- `../REPRODUCE.md` — end-to-end runbook
- `../REFERENCE_RESULTS.md` — the numbers
- `../NEXT_SESSION.md` — state, gotchas, open items, file anchors

Use this raw log when you need the *why* behind a decision or the detail a
diagnosis hinged on — the curated docs deliberately omit the journey.

## Note
Scanned clean of secrets before commit (the SFT-checkpoint HF token and the
GitHub PAT used during the session were deliberately never written to memory).
Some chronological lines reference `/tmp/...` paths and run IDs that no longer
exist post-teardown — they are historical record, not live pointers.
