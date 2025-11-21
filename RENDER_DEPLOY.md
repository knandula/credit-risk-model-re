# Render Deployment Instructions

## Deploy to Render (Recommended for Dash Apps)

Render is free and works better with Python web apps than Vercel.

### Steps:

1. **Go to Render**: https://render.com (sign up with GitHub)

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `knandula/credit-risk-model-re`

3. **Configure the service**:
   - **Name**: `credit-risk-model-re` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn dashboard:server --bind 0.0.0.0:$PORT`
   - **Instance Type**: `Free`

4. **Click "Create Web Service"**

Your app will be live at: `https://credit-risk-model-re.onrender.com`

**Note**: Free tier may spin down after inactivity, taking 30-60 seconds to wake up on first request.

---

## Alternative: PythonAnywhere (Also Free)

1. Go to https://www.pythonanywhere.com (free account)
2. Upload your files
3. Configure WSGI app with Flask
4. Your app: `https://yourusername.pythonanywhere.com`

---

## Alternative: Heroku (Paid, but reliable)

1. Install Heroku CLI
2. Run:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

Your app: `https://your-app-name.herokuapp.com`
