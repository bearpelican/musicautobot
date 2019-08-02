import os
from api import app

if __name__ == "__main__":
    app.run()

# To Run:
# yarn build
# gunicorn -w 8 run_guni:app -b 127.0.0.1:5000
