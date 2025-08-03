# locustfile.py

from locust import HttpUser, task, between

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def access_ui(self):
        # This simply tests if the main UI page is responsive
        self.client.get("/")