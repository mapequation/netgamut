# netgamut

## Start new work in its own worktree

Start each new piece of work in a git worktree under `.claude/worktrees/` in
this folder, not on a new branch in the main checkout. Then several sessions can
work in parallel without touching each other's files. Skip this when the session
already runs in a worktree, as one started with the app's Worktree box does.

Create it with git, from the local branch the work builds on, usually `main`:

```bash
git worktree add .claude/worktrees/<name> -b <name> main
```

Keep the session in this folder rather than moving it into the worktree (so not
the `EnterWorktree` tool): edit the worktree's files by their paths, and run
commands there with `git -C .claude/worktrees/<name> …` or
`(cd .claude/worktrees/<name> && …)`.

The main checkout may hold other sessions' uncommitted changes: leave those
alone, and commit only your own files. `.claude/worktrees/` is gitignored.
