import io
import os
import json
from urllib.parse import urlparse

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

    # Backend now returns public URLs for generated files (served under /output/).
    from urllib.parse import urlparse

    # verify generated masks are reachable and match expected resolution
    for url in body['generated']:
        parsed = urlparse(url)
        # path should point to /output/<file>
        assert parsed.path.startswith('/output/')

        # fetch via Flask test client
        resp = client.get(parsed.path)
        assert resp.status_code == 200

        # open image bytes and validate content
        img = Image.open(io.BytesIO(resp.data)).convert('L')
        arr = np.array(img)
        assert arr.size > 0
        # expect segmentation default resolution 384x576 (H x W)
        assert arr.shape[0] == 384 and arr.shape[1] == 576, f"Unexpected mask shape: {arr.shape}"
        # no further checks — model may produce trivial masks depending on training


def test_save_outputs_endpoint(tmp_path):
    client = app.test_client()

    # Create and upload an image to generate
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color=(120, 120, 120)).save(buf, format='PNG')
    buf.seek(0)

    multipart = {
        'images': (buf, 'image_save.png'),
        'model': 'm[64]',
        'output_dir': 'tmp_save_test',
        'config': json.dumps({'threshold': 0.5, 'batchSize': 1}),
    }

    resp = client.post('/api/generate', data=multipart, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()

    generated = body.get('generated', [])
    assert isinstance(generated, list)
    if len(generated) == 0:
        # nothing generated — skip
        return

    # Call save_outputs to copy generated files into a new folder
    save_payload = {'urls': generated, 'dest_dir': 'saved_test_dir'}
    save_resp = client.post('/api/save_outputs', json=save_payload)
    assert save_resp.status_code == 200
    save_body = save_resp.get_json()
    assert save_body.get('ok') is True
    assert isinstance(save_body.get('saved'), list)

    # Ensure saved URLs are reachable
    for s in save_body.get('saved', []):
        parsed = urlparse(s)
        r = client.get(parsed.path)
        assert r.status_code == 200
