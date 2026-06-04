# Deployment

Two pieces: (1) Hugging Face Space runs the actual Streamlit app, (2) Vercel owns your custom subdomain and forwards to the HF Space.

## Step 1 — Deploy to Hugging Face Spaces (5 min)

### 1a. Create the Space

1. Go to https://huggingface.co/new-space
2. Sign up / sign in (free).
3. Fill in:
   - **Owner**: your HF username (e.g., `julieluan`)
   - **Space name**: `ai-trading-sim` (or whatever you want)
   - **License**: MIT
   - **Space SDK**: **Streamlit**
   - **Hardware**: CPU basic · free
   - **Public**
4. Click **Create Space**.

### 1b. Get an HF write token

1. Go to https://huggingface.co/settings/tokens
2. **New token** → name "deploy" → role **write** → Create.
3. Copy the token (`hf_…`). You'll need it once.

### 1c. Push from this repo

From your terminal, in `/Users/ruijialuan/Desktop/stock-prediction`:

```bash
# Replace HF_USER and HF_TOKEN with your values
git remote add hf https://HF_USER:HF_TOKEN@huggingface.co/spaces/HF_USER/ai-trading-sim
git push hf main
```

HF will start building automatically. Watch the build log at your Space URL (`https://huggingface.co/spaces/HF_USER/ai-trading-sim`). First build takes ~3 minutes.

When the status turns **Running**, your app is live at:
```
https://HF_USER-ai-trading-sim.hf.space
```

## Step 2 — Custom subdomain via Vercel (10 min)

You wanted `pit.yourcompany.com` (or whatever) on Vercel pointing at the app. The cleanest pattern:

- Vercel owns your custom subdomain + SSL.
- A tiny `vercel.json` rewrites every request to the HF Space URL.
- WebSockets pass through transparently (Streamlit needs them).

### 2a. Make a tiny Vercel project

```bash
mkdir ~/Desktop/trading-game-proxy
cd ~/Desktop/trading-game-proxy

# Create the only file you need:
cat > vercel.json << 'EOF'
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "https://HF_USER-ai-trading-sim.hf.space/$1"
    }
  ]
}
EOF

git init -b main
git add vercel.json
git commit -m "Vercel proxy to HF Space"
gh repo create julieluan/trading-game-proxy --public --source=. --remote=origin --push
```

(Replace `HF_USER` with your actual HF username.)

### 2b. Deploy to Vercel

1. Go to https://vercel.com/new
2. Import `julieluan/trading-game-proxy`.
3. Click **Deploy**. (No build needed — it's just a config file.)

### 2c. Bind your custom subdomain

1. In the Vercel project → **Settings** → **Domains**.
2. Add `pit.yourcompany.com` (or your chosen subdomain).
3. Vercel shows the DNS record you need to add at your domain registrar:
   - Type: `CNAME`
   - Name: `pit` (or whatever subdomain)
   - Value: `cname.vercel-dns.com.`
4. Add that record at your DNS provider. Wait ~5 min for propagation.
5. Vercel auto-issues SSL cert via Let's Encrypt.

When the green check appears, **`pit.yourcompany.com` → HF Space → your Streamlit app**. WebSockets work, browser only sees your custom domain.

## Subsequent updates

After this is set up, deploying new versions is just:

```bash
# In the main stock-prediction repo:
git add . && git commit -m "..."
git push origin main          # pushes to GitHub
git push hf main              # pushes to HF Space → auto-rebuilds
```

The Vercel project doesn't need re-deployment — it always rewrites to whatever's currently at the HF Space URL.

## Troubleshooting

- **HF build fails on `yfinance`**: yfinance hits Yahoo's API. If HF blocks outbound calls, the chart will fall back to whatever's cached. If this becomes a problem, we can switch to bundling a static CSV (one-shot pre-fetch).
- **Vercel proxy 504**: HF Space is asleep or rebuilding. Visit the HF URL directly — once it's running, the proxy works.
- **WebSocket disconnect**: if you see "connection lost" in the Streamlit UI, the Vercel free tier may be throttling. Upgrade Vercel project to Pro ($20/mo) OR switch to a 301 redirect (less elegant but cheaper).
