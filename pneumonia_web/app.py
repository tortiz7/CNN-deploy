from flask import Flask, render_template, request
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Number of items per page
ITEMS_PER_PAGE = 50

@app.route('/')
def index():
    # Get the current page number from query parameters, default to 1
    page = request.args.get('page', 1, type=int)
    
    # Get all keys from Redis
    keys = r.keys('*')
    total_keys = len(keys)

    # Calculate pagination
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    keys_to_display = keys[start:end]

    # Get the data for the keys to display
    data = {}
    for key in keys_to_display:
        data[key] = r.hgetall(key)

    return render_template('index.html', data=data, page=page, total_keys=total_keys, items_per_page=ITEMS_PER_PAGE)

if __name__ == '__main__':
    app.run(debug=True)
