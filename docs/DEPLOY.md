# Put it on GitHub and make it live

## 0. What you need on your computer
Only **git** (and a GitHub account). Training runs on GitHub's servers via the "Train release" workflow, and hosting runs on a free cloud service. You do not need to install RDKit or train locally. (You *can*: `make setup data train`.)

## 1. Upload to GitHub
```bash
cd aromascope
git init -b main
git add .
git commit -m "AromaScope: initial project"
# create an EMPTY repo on github.com (no README), then:
git remote add origin https://github.com/<you>/aromascope.git
git push -u origin main
```
CI starts automatically. The first run installs RDKit on GitHub's machines and runs the full test suite, including a train -> release -> API test.

## 2. Train the real model (no local setup)
GitHub repo -> **Actions** -> **Train release** -> **Run workflow** (version `0.1.0`). It downloads the data, trains, evaluates and commits `artifacts/releases/0.1.0/` plus `CURRENT` to `main` (about 5 to 15 minutes). Then `git pull` if you work locally.
Read `artifacts/releases/0.1.0/model_card.md` and `evaluation_report.json`: these are your real results.

## 3. Deploy (single service: API + site together)
**Recommended: Hugging Face Spaces (free, 16 GB RAM, built for ML demos)**
1. huggingface.co -> New Space -> SDK **Docker** -> Blank.
2. In the Space, add a `README.md` whose first lines are:
   ```
   ---
   title: AromaScope
   sdk: docker
   app_port: 7860
   ---
   ```
   (keep this README only in the Space, not on GitHub), then push the repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/aromascope
   git push space main --force
   ```
3. It builds the Dockerfile and serves at `https://<you>-aromascope.hf.space`. Check `/health/ready`.

**Alternative: Render** (free web service; sleeps when idle). New -> Web Service -> connect the GitHub repo -> Runtime **Docker** -> deploy. Set `PORT=7860` if asked.

## 4. Optional: host the site separately (Vercel / Netlify / GitHub Pages)
Deploy the `apps/web` folder as a static site and set your API address in `apps/web/config.js` (`apiBase`). On the API set `AROMASCOPE_ORIGINS=https://your-site.example`.

## Environment variables
| Variable | Default | Meaning |
|---|---|---|
| `AROMASCOPE_ORIGINS` | localhost | Comma-separated allowed browser origins (CORS) |
| `AROMASCOPE_RATE_LIMIT` | 60 | Requests per minute per client (0 disables) |
| `AROMASCOPE_RELEASE_DIR` | latest | Pin a specific release folder |
| `AROMASCOPE_TRUST_PROXY` | 1 in Docker | Use `X-Forwarded-For` for rate limiting |

## Before going public
- Review the Leffingwell terms (see `docs/dataset_card.md`) and decide `redistribution_approved`.
- Keep the in-app limitations visible: it predicts from structure, it is not a sensor.
