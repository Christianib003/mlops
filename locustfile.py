# locustfile.py
from locust import HttpUser, task, between
import os

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_image(self):
        # Make sure you have a 'test_image.jpg' in the same folder
        image_path = "test_image.jpg"
        if not os.path.exists(image_path):
            return

        with open(image_path, "rb") as image_file:
            self.client.post(
                "/predict/",
                files={"file": (os.path.basename(image_path), image_file, "image/jpeg")}
            )