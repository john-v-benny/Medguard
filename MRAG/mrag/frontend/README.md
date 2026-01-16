# MRAG Frontend

This is a small React app scaffold using Vite. It contains `src/App.jsx` with the `MedicalMRAGApp` component.

Run locally:

```bash
cd mrag/frontend
npm install
npm run dev
```

The app calls the backend at `VITE_API_URL` if set, otherwise `http://localhost:8000`. Example:

```bash
# run dev with custom backend url
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

Notes:
- The component uses Tailwind-like classes. I included minimal CSS so it looks reasonable without Tailwind. If you want Tailwind, I can install and configure it.
- The backend endpoint expected is `POST /mrag` which should accept `{ symptoms: string[], bayesian_output: object }` and return JSON with an `explanation` field. If you have a different endpoint, I can update the default URL or add a proxy.
