from gradio_client import Client

client = Client("wilsonchang17/scamshield-api")
result = client.predict(
		message="Hello!!",
		api_name="/predict"
)
print(result)