import requests

class SandboxFusionClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def get_execution_result(self, code, language):
        url = f"{self.base_url}/run_code"
        payload = {
            "code": code,
            "language": language
        }
        response = requests.post(url, json=payload)
        return response.json()


if __name__ == "__main__":
    client = SandboxFusionClient("http://localhost:8080")
    print(client.get_execution_result("print('Hello, World!')", "python"))
