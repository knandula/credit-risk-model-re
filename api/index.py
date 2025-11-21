#!/usr/bin/env python3
"""
Vercel Serverless Function Wrapper for Dashboard
================================================

This is a lightweight wrapper that imports the Dash app server.
Note: May still face size limits with scientific libraries.
"""

import sys
import os

# Add parent directory to path to import dashboard
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import server as application

# Vercel expects 'app' or 'application' variable
app = application
