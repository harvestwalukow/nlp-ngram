# Flask Application

A basic Flask web application with a modern UI and API endpoints.

## Features

- 🚀 Basic Flask application structure
- 🎨 Modern, responsive UI with CSS Grid and Flexbox
- 📱 Mobile-friendly design
- 🔌 RESTful API endpoint for health checks
- 📁 Organized project structure

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

## Development

- The app runs in debug mode by default
- Static files are served from the `static/` directory
- Templates are located in the `templates/` directory
- API endpoints are defined in `app.py`

## API Endpoints

- `GET /` - Main page
- `GET /api/health` - Health check endpoint (returns JSON)

## Customization

You can easily extend this application by:

- Adding new routes in `app.py`
- Creating new templates in `templates/`
- Adding custom CSS in `static/css/`
- Implementing JavaScript functionality in `static/js/`
