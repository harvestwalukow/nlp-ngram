## Project Structure

```
n-gram/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/         # HTML templates
│   └── index.html
└── static/           # Static files (CSS, JS, images)
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python app.py
   ```

3. **Access the application:**
   - Open your browser and go to `http://localhost:5000`
   - The API health check endpoint is available at `http://localhost:5000/api/health`
