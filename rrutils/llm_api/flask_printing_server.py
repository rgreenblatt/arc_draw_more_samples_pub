from flask import Flask, jsonify, request

app = Flask(__name__)

# Global variable to store received messages
received_messages = []

@app.route('/', defaults={'path': ''}, methods=['POST'])
@app.route('/<path:path>', methods=['POST'])
def catch_all(path):
    print(f"Path: {path}")
    json_data = request.get_json()
    json_data["path"] = path
    print(f"JSON Data: {json_data}")
    
    # Store the received message
    if json_data is not None:
        received_messages.append(json_data)
        
    return jsonify({"message": "Request received"})

@app.route('/get_messages', methods=['GET'])
def get_messages():
    global received_messages
    
    # Return the received messages and empty the list
    response = jsonify(received_messages)
    received_messages = []
    
    return response

if __name__ == '__main__':
    app.run(debug=False, port=8944)