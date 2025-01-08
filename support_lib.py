import re

def clean_response(response):
    response = re.sub(r'\n', '', response)
    response = re.sub(r'\"', '', response)
    response = re.sub(r'`', '', response)
    return response