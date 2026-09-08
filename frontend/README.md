# Crochet Pattern Converter Frontend

Next.js App Router frontend for uploading crochet-chart PDFs and editing the written pattern returned by the FastAPI service.

## Setup

```bash
npm install
```

Copy `.env.example` to `.env.local` and set the backend URL:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the frontend from this directory:

```bash
npm run dev
```

Open `http://localhost:3000`. The FastAPI backend must also be running for conversion requests.

## Checks

```bash
npm run lint
npm run build
```

The interface uses Tailwind CSS and the self-hosted Gaegu font through `next/font`.
