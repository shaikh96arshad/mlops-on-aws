import os
import io
import boto3
import json
import csv

# grab environment variables 
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = int(result['predictions'][0]['predicted_label'])
    
    if(pred == 0):
        return 'DERMASON'
    if(pred == 1):
        return 'SIRA'
    if(pred == 2):
        return 'SEKER'
    if(pred == 3):
        return 'HOROZ'
    if(pred == 4):
        return 'CALI'
    if(pred == 5):
        return 'BARBUNYA'
    if(pred == 6):
        return 'BOMBAY'