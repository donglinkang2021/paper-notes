# Paper Notes — Quartz Deployment Plan

## Overview

Deploy `learn-claude-code` knowledge base (paper summaries + MOC indexes + cross-paper insights) as a Quartz v4 static site at `donglinkang2021.github.io/paper-notes`.

## Architecture

- **Content source**: `learn-claude-code` repo (maintained in Obsidian/VSCode)
- **Build repo**: `paper-notes` (Quartz v4 clone, compiles content to static site)
- **Deploy**: GitHub Actions → GitHub Pages

## Content Structure

```
content/
├── index.md          # Homepage with links to areas & insights
├── knowledge/        # Per-paper summaries (summary_*.md)
├── areas/            # Map of Content files (research domain groupings)
└── insights/         # Cross-paper thematic insights
```

## Key Config

| File | Key Changes |
|------|-------------|
| `quartz.config.ts` | pageTitle="Paper Notes", baseUrl="donglinkang2021.github.io/paper-notes", locale="zh-CN" |
| `quartz.layout.ts` | Footer links updated to personal GitHub/Email/Home |
| `.github/workflows/deploy.yml` | Triggers on `v4` branch push |

## Daily Workflow

```bash
# 1. Sync content from source repo
./sync-content.sh

# 2. Local preview
npx quartz build --serve

# 3. Push to deploy
npx quartz sync
```
