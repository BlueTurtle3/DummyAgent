import os
from huggingface_hub import InferenceClient
from utils.utils import SYSTEM_PROMPT, LOCATION

from dotenv import load_dotenv


def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"


def main():
    try:
        print("main.py started")
        load_dotenv()
        HF_TOKEN = os.environ.get("HF_TOKEN")
        client = InferenceClient(model="moonshotai/Kimi-K2.5", token=HF_TOKEN)
        # 1. First response with the system prompt and user question, stopping at "Observation:"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What's the weather in {LOCATION}?"},
        ]
        output = client.chat.completions.create(
            messages=messages,
            stop=["Observation:"],
            max_tokens=150,
            extra_body={"thinking": {"type": "disabled"}},
        )
        # 1.1 print the first response, which should include the Thought and Action with the JSON blob for get_weather
        print(output.choices[0].message.content)
        # 2. Append the Observation with the result of get_weather and send it back to the model, asking for the final 
        # answer. This should trigger the model to provide the final answer based on the observation.
        messages.append(
            {
                "role": "assistant",
                "content": output.choices[0].message.content
                + "Observation:\n"
                + get_weather(LOCATION),
            },
        )
        output = client.chat.completions.create(
            messages=messages,
            stream=False,
            max_tokens=200,
            extra_body={'thinking': {'type': 'disabled'}},
        )
        print(output.choices[0].message.content)
        print("main.py ended")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
