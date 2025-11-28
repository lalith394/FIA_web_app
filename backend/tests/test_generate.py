import io
import os
import json

from app import app


def test_generate_endpoint(tmp_path):
    client = app.test_client()

    # Create a small in-memory file for upload
    data = {
        'model': 'm[64]',
        'output_dir': 'outputs/test',
        'config': json.dumps({'threshold': 0.5, 'batchSize': 1})
    }

    # Create small valid PNG images in-memory so PIL can open them during the test
    from PIL import Image
    img1 = Image.new('RGB', (64, 64), color=(10, 20, 30))
    buf1 = io.BytesIO()
    img1.save(buf1, format='PNG')
    buf1.seek(0)

    img2 = Image.new('RGB', (64, 64), color=(50, 120, 200))
    buf2 = io.BytesIO()
    img2.save(buf2, format='PNG')
    buf2.seek(0)

    file1 = (buf1, 'image1.png')
    file2 = (buf2, 'subdir/image2.png')

    # Build data as the Flask test client expects: a dict where a field's value
    # can be a list for multiple files under the same key.
    multipart_data = {
        'images': [file1, file2],
        'model': data['model'],
        'output_dir': data['output_dir'],
        'config': data['config'],
    }

    response = client.post('/api/generate', data=multipart_data, content_type='multipart/form-data')

    assert response.status_code == 200
    body = response.get_json()
    assert body['ok'] is True
    assert 'saved' in body
    assert isinstance(body['saved'], list)
    # Generated masks should be returned for segmentation models
    assert 'generated' in body
    assert isinstance(body['generated'], list)

    # Generated mask files should exist on disk and not be all-white
    from PIL import Image
    import numpy as np

    for path in body['generated']:
        # generated paths returned are full paths written by the backend
        assert os.path.exists(path)

        # ensure mask is not all-white (all 255) and not all-black
        img = Image.open(path).convert('L')
        arr = np.array(img)
        assert arr.size > 0
        assert not np.all(arr == 255), "Generated mask is all white"
        assert not np.all(arr == 0), "Generated mask is all black"
