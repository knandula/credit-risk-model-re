# Real Estate Credit Risk Model Dashboard

Interactive Monte Carlo simulation dashboard for analyzing real estate-backed credit products.

## Features

- 8 adjustable parameters for scenario analysis
- Real-time Monte Carlo simulation
- 6 interactive charts showing risk metrics
- Responsive design with adjustable panels
- Comprehensive risk modeling (BGM interest rates, GBM collateral, default modeling)

## Deployment to Vercel

### Prerequisites
1. Install Vercel CLI: `npm install -g vercel`
2. Sign up for a free account at https://vercel.com

### Deploy Steps

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy from this directory**
   ```bash
   vercel
   ```
   
   Follow the prompts:
   - Set up and deploy? **Y**
   - Which scope? Select your account
   - Link to existing project? **N**
   - Project name: `realestate-model-simulation` (or your choice)
   - In which directory is your code located? **./** (current directory)

3. **Production deployment**
   ```bash
   vercel --prod
   ```

### Alternative: Deploy via Vercel Dashboard

1. Go to https://vercel.com/new
2. Import your git repository
3. Vercel will auto-detect the configuration from `vercel.json`
4. Click "Deploy"

## Local Development

```bash
pip install -r requirements.txt
python dashboard.py
```

Then open http://127.0.0.1:8050/ in your browser.

## Important Notes

⚠️ **This is for educational purposes only. Not investment advice.**

See the disclaimer section in the dashboard for full details about model limitations.
