# AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An **ATS score out of 100**, built from five weighted categories
  (formatting, keywords, impact, clarity, relevance)
- **Specific, prioritised improvements** for your resume
- **Found vs. missing keywords** (paste a job description for a sharper match)
- **Before/after bullet rewrites**
- Instant **rule-based checks** (email, phone, section headings, length, metrics)
- A downloadable Markdown report

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

## How the score works

Gemini rates each category from 0 to 100 using a strict rubric. The app then
combines them with fixed weights (formatting 25%, keywords 25%, impact 20%,
clarity 15%, relevance 15%), so the overall number is calculated by the app,
not guessed by the AI. Treat it as a guide: real ATS products each score
differently, and no tool can reproduce them exactly.

## Run locally

1. Install Python 3.10 or newer.
2. Get a free Gemini API key at https://aistudio.google.com/apikey
3. In the project folder:

```bash
pip install -r requirements.txt
```

4. Give the app your key. Choose **one** way:

   - Type it into the sidebar when the app opens, or
   - Create `.streamlit/secrets.toml`:

     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```

   - Or set an environment variable: `GEMINI_API_KEY=your-key-here`

5. Start the app:

```bash
streamlit run app.py
```

## Choosing the Gemini model

The default is `gemini-3.8-flash`. Google retires and releases models often, so
if you see a "model not found" error, change the model name in the sidebar, or
set `GEMINI_MODEL` in your secrets. The current list is at
https://ai.google.dev/gemini-api/docs/models

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

5. Click **Deploy**.

**Never commit your API key to GitHub.** Keep `.streamlit/secrets.toml` out of
the repository (add it to `.gitignore`) and use the Secrets box on Streamlit.

## Privacy

The text of the uploaded resume is sent to Google's Gemini API for analysis.
The app does not store files; results live only in your browser session.

## Limitations

- Scanned (image-only) PDFs cannot be read. If a parser can't read your file,
  neither can an ATS. Export a text-based PDF or DOCX.
- The AI can make mistakes. Don't add keywords for skills you don't have.
- Files are limited to 5 MB; very long resumes are truncated to about 20,000 characters.
