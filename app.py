from flask import Flask, jsonify, request
from flask_cors import CORS
from breed import run_breed

app = Flask(__name__)
CORS(app)

@app.route('/breed', methods=['POST'])
def breed() :
    body = request.get_json();
    images = body['images'];
    
    if images :
        image = images[0]; #일단은 첫번째 이미지로만 분석
        path = image['location']
        result = run_breed(path)
        
    return jsonify(
        breed=result,
    )

if __name__ == '__main__':
    app.run(debug=True)