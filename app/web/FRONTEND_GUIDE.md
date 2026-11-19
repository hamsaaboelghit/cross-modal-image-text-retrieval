# VisionMatch AI — Frontend

Responsive Next.js interface for the cross-modal image–text retrieval API.

## Stack

- Next.js App Router
- React + TypeScript
- Tailwind CSS
- lucide-react icons

## Routes

- `/` — home page, Text-to-Image and Image-to-Text search, upload preview, ranked results.
- `/compare` — reads saved evaluation summaries from the backend `/metrics` endpoint.
- `/about` — project explanation, model pipeline, and dataset overview.

## Configure and run

Copy `.env.local.example` to `.env.local` in this directory and set the backend URL:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Install dependencies and run the development server:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## API integration

The frontend calls:
- `GET /models` to discover the configured model.
- `POST /search/text` with JSON `{ query, top_k, model }`.
- `POST /search/image?top_k=...&model=...` with multipart `file`.
- `GET /metrics` for real saved model-evaluation data.
- `GET /images/{image_name}` to render gallery image results.

Search states include loading, validation and API errors, empty results, result ranking, similarity scores, and elapsed time. Uploaded image previews are local browser object URLs; the file is sent only when the user submits an image search.

The backend process currently serves one configured model. The model selector reflects the model returned by that API process; it does not imply that every architecture has a trained checkpoint available.

## Important

The frontend requires a running backend with a compatible trained checkpoint, vocabulary, and retrieval gallery. The model comparison page intentionally displays an unavailable state until a real evaluation summary is configured. No demo scores or synthetic retrieval results are presented as real model output.
