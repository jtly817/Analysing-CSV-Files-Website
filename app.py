from flask import Flask
from routes import main_bp
from services.database import get_connection

import glob
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['DATABASE_CONNECTION'] = get_connection()
app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(debug=True)

