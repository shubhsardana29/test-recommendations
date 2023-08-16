from flask import Flask, render_template, request, jsonify
import joblib
from recommendation_logic import generate_hybrid_recommendations

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()  # Get JSON data
        user_id = int(data['user_id'])
        num_recommendations = int(data['num_recommendations'])
        
        hybrid_recommendations = generate_hybrid_recommendations(user_id, num_recommendations)
        
        # Convert int64 values to Python integers
        hybrid_recommendations = [(int(product_id), float(score)) for product_id, score in hybrid_recommendations]
        user_id = int(user_id)
        
        # Create JSON response
        response_data = {
            "recommendations": hybrid_recommendations,
            "user_id": user_id
        }
        
        return jsonify(response_data)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
