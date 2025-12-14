import time
import requests
# Function to measure response time
def measure_response_time(endpoint, data):
    start_time = time.time()
    response = requests.post(endpoint, data=data)
    end_time = time.time()
    response_time = end_time - start_time
    return response_time
# Measure response time for sentiment analysis
data = {'text': 'This is a great product!', 'model_type': 'logistic_regression'}
response_time = measure_response_time('http://localhost:5000/analyze', data)
print(f'Sentiment Analysis Response Time: {response_time} seconds')