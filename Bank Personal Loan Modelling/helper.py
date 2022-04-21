class myClass1(object):
    def get_s3_object(bucket,key):
        s3 = b3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=Key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df