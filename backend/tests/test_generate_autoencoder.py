import io
import os
import sys
from urllib.parse import urlparse
import json

# Ensure backend package root is importable when pytest is invoked from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app


def test_generate_autoencoder(tmp_path):
    client = app.test_client()

    model_name = 'ae_e6_d6_w192_h128_rfmid_sk[d6]_color_[LATEST]_sig'

    img = io.BytesIO()
    from PIL import Image
    Image.new('RGB', (64, 64), color=(80, 90, 120)).save(img, format='PNG')
    img.seek(0)

    multipart = {
        'images': (img, 'auto_demo.png'),
        'model': model_name,
        'output_dir': 'tmp_auto_test',
        'config': json.dumps({'threshold': 0.5, 'batchSize': 1}),
    }

    resp = client.post('/api/generate', data=multipart, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    # show what returned so we can assert conditions
    assert 'generated' in body
    print('RESPONSE BODY:', body)
    # The API should run inference for this autoencoder model and return image URLs
    assert isinstance(body['generated'], list)
    # If any URLs returned, verify they are under /output/ and reachable via the test client
    from urllib.parse import urlparse
    for url in body['generated']:
        parsed = urlparse(url)
        assert parsed.path.startswith('/output/')
        r = client.get(parsed.path)
        assert r.status_code == 200


def test_generate_autoencoder_with_absolute_outdir(tmp_path):
    """If a client supplies an absolute output_dir, the API should normalize it
    and still save outputs under the backend's output/ directory so URLs are reachable.
    """
    client = app.test_client()

    model_name = 'ae_e6_d6_w192_h128_rfmid_sk[d6]_color_[LATEST]_sig'

    img = io.BytesIO()
    from PIL import Image
    Image.new('RGB', (64, 64), color=(80, 90, 120)).save(img, format='PNG')
    img.seek(0)

    abs_dir = str(tmp_path.joinpath('abs_output_dir'))

    multipart = {
        'images': (img, 'auto_demo_abs.png'),
        'model': model_name,
        'output_dir': abs_dir,  # absolute path supplied by client
        'config': json.dumps({'threshold': 0.5, 'batchSize': 1}),
    }

    resp = client.post('/api/generate', data=multipart, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'generated' in body
    assert isinstance(body['generated'], list)

    # generated files should be reachable under /output/ served by the backend
    for url in body['generated']:
        parsed = urlparse(url)
        assert parsed.path.startswith('/output/')
        r = client.get(parsed.path)
        assert r.status_code == 200
