from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# Config
API_URL = 'http://backendapi:5000' 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/table')
def table():
    try:
        page = request.args.get('page', 1, type=int)
        response = requests.get(f'{API_URL}/predictions', params={'page': page})
        response.raise_for_status() 
        data = response.json()
        
        return render_template('partials/table.html', **data)
    
    except requests.RequestException as e:
        app.logger.error(f"API request failed: {str(e)}")
        return render_template('partials/error.html', 
                             message="Unable to fetch data from API"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)