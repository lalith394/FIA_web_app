import io
import os
import json

from app import app


def test_generate_endpoint(tmp_path):
    client = app.test_client()

    # Create a small in-memory file for upload
    data = {
        'model': 'test-model',
        'output_dir': 'outputs/test',
        'config': json.dumps({'threshold': 0.9, 'batchSize': 1, 'preprocessing': 'Normalize Only'})
    }

    file1 = (io.BytesIO(b"hello world"), 'image1.png')
    file2 = (io.BytesIO(b"another image"), 'subdir/image2.png')

    response = client.post('/api/generate', data={
        'images': [file1, file2],
        **data
    }, content_type='multipart/form-data')

    assert response.status_code == 200
    body = response.get_json()
    assert body['ok'] is True
    assert 'saved' in body
    assert isinstance(body['saved'], list)
