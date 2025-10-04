import sys
import os

# Ensure the project directory is on sys.path
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Dash app (dashboard.py must expose `app`)
from dashboard import app

# For WSGI servers use the Flask server contained in the Dash app
application = app.server
