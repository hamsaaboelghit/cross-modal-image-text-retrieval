# VisionMatch AI — Frontend

Phase 8 frontend workspace for the Cross-Modal Image-Text Retrieval project.

## Stack

- Next.js App Router
- React + TypeScript
- Tailwind CSS
- lucide-react icons

## Routes

- `/` — landing page and interactive retrieval input (text-to-image / image-to-text)
- `/compare` — experiment overview and retrieval-metrics placeholders
- `/about` — project explanation, model pipeline, and dataset overview

## Run locally

From this directory:

```bash
npm install
npm run dev
```

Open the local URL printed by Next.js (normally `http://localhost:3000`).

## Current integration status

This is a frontend-only preview. The search form validates text/image input and presents a clear preview state; it does not call a backend or claim to return retrieval results.

The comparison page intentionally shows empty metric slots. Do not replace them with illustrative scores: populate them only from the project's saved evaluation outputs after training and evaluation have completed.

Image upload currently accepts image MIME types up to 10 MB and creates a local preview in the browser. No uploaded image is sent to a server.

## Before backend integration

1. Agree on API request/response schemas.
2. Connect the search action to the inference endpoint.
3. Render returned image/caption matches and similarity scores.
4. Connect experiment metrics to the saved evaluation artifacts.
5. Add loading, empty, and error states for real API responses.
