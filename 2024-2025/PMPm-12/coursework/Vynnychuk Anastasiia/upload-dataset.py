import boto3

client = boto3.client('s3', **{
    "endpoint_url": 'https://s3-maga.fra1.digitaloceanspaces.com',
    "aws_access_key_id": 'DO8017EFRVXJKWUEHBFG',
    "aws_secret_access_key": '3zGGSExDTJE5BNL8I3B5mxvHCEtMn/LwYp3345dIpp8'
})

client.upload_file('./data/dfg-ultralytics.zip', 's3-maga', 'dfg-ultralytics.zip')
