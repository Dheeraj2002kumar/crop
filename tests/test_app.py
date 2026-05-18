import os
import unittest
import json
from PIL import Image
from io import BytesIO
import numpy as np

# Set environment variable to suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import app as flask_app
import src.config as config

class TestBioShieldWebPortal(unittest.TestCase):

    def setUp(self):
        # Configure Flask app for testing
        flask_app.app.config['TESTING'] = True
        self.client = flask_app.app.test_client()

    def test_home_page_renders_successfully(self):
        """Test that the main diagnostic interface loads with correct title and elements."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        html_content = response.data.decode('utf-8')
        self.assertIn('<title>BioShield - Smart Crop Disease Diagnostic Portal</title>', html_content)
        self.assertIn('BioShield AI', html_content)
        self.assertIn('Leaf Scan Analyzer', html_content)
        self.assertIn('Quick Diagnostic Samples', html_content)
        self.assertIn('Awaiting Analysis', html_content)

    def test_predict_sample_valid_class(self):
        """Test that predict-sample endpoint successfully returns a rich diagnostic report for a valid class."""
        # We test with "Tomato_healthy"
        response = self.client.get('/predict-sample/Tomato_healthy')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('class', data)
        self.assertIn('display_name', data)
        self.assertIn('confidence', data)
        self.assertIn('status', data)
        self.assertIn('type', data)
        self.assertIn('severity', data)
        self.assertIn('symptoms', data)
        self.assertIn('treatment', data)
        
        # Verify specific details of the diagnosis
        self.assertIn(data['class'], flask_app.class_names)
        self.assertIn('status', data)
        self.assertIsInstance(data['symptoms'], list)

    def test_predict_sample_invalid_class_returns_400(self):
        """Test that predict-sample endpoint safely rejects invalid agricultural classes."""
        response = self.client.get('/predict-sample/Invalid_Plant_Disease')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('error', data)
        self.assertIn('Invalid class name', data['error'])

    def test_predict_upload_endpoint(self):
        """Test uploading a dynamically generated leaf image to the live predict endpoint."""
        # Create a dummy RGB image to simulate a plant leaf
        img = Image.new('RGB', (224, 224), color=(34, 139, 34)) # Forest Green
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Post to /predict
        response = self.client.post(
            '/predict',
            data={'image': (img_byte_arr, 'test_leaf.jpg')},
            content_type='multipart/form-data'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        
        # Verify complete JSON output structure
        self.assertIn('class', data)
        self.assertIn('display_name', data)
        self.assertIn('confidence', data)
        self.assertIn('status', data)
        self.assertIn('treatment', data)
        self.assertIn('symptoms', data)
        
        print(f"\n[+] Auto-Test verified prediction successful for upload!")
        print(f"    Class predicted: {data['class']} | Confidence: {data['confidence']}")

    def test_get_sample_thumbnail(self):
        """Test retrieving local sample thumbnails successfully."""
        response = self.client.get('/get-sample-thumbnail/Tomato_healthy')
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.content_type, ['image/jpeg', 'image/png', 'image/svg+xml'])

if __name__ == '__main__':
    unittest.main()
