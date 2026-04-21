import requests

url = "https://yinli.one/v1/chat/completions"
body = """{
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "string"
    }
  ]
}"""
response = requests.request("POST", url, data = body, headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer sk-m8DX1lUuiSboZxOfLLEOdDULWkQHcBqdgeKbU0Ex28TZ73sf"
})

print(response.text)