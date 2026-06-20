# OpenAI Python SDK

> `pip install openai` · Python 3.9+

---

## Table of Contents

- [Client Initialization](#client-initialization)
- [Chat Completions](#chat-completions)
- [Structured Outputs](#structured-outputs)
- [Responses API](#responses-api)
- [Streaming](#streaming)
- [Async Usage](#async-usage)
- [Tool / Function Calling](#tool--function-calling)
- [Vision (Image Input)](#vision-image-input)
- [Embeddings](#embeddings)
- [Images (DALL-E)](#images-dall-e)
- [Audio](#audio)
- [Files](#files)
- [Fine-tuning](#fine-tuning)
- [Batch](#batch)
- [Moderations](#moderations)
- [Models](#models)
- [Error Handling](#error-handling)
- [Pagination](#pagination)
- [Realtime API (WebSocket)](#realtime-api-websocket)
- [Utilities](#utilities)

---

## Client Initialization

### `class openai.OpenAI(api_key, timeout, max_retries)`

Synchronous client for the OpenAI API — reads `OPENAI_API_KEY` from the environment by default.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str` | `None` | Explicit API key; if omitted, reads `OPENAI_API_KEY` from environment |
| `timeout` | `float` | `60.0` | Per-request timeout in seconds |
| `max_retries` | `int` | `2` | Number of automatic retries on transient errors |

```python
from openai import OpenAI

# 1. Default init — reads OPENAI_API_KEY from environment
client = OpenAI()

# 2. Explicit API key
client = OpenAI(api_key="sk-...")

# 3. Custom timeout and retry settings
client = OpenAI(timeout=20.0, max_retries=3)

# 4. Per-call override without modifying the original client
client2 = client.with_options(timeout=5.0, max_retries=0)
```

---

### `class openai.AsyncOpenAI(api_key, timeout, max_retries)`

Drop-in async version of `OpenAI` — every method must be called with `await`.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `api_key` | `str` | `None` | Explicit API key; if omitted, reads `OPENAI_API_KEY` from environment |
| `timeout` | `float` | `60.0` | Per-request timeout in seconds |
| `max_retries` | `int` | `2` | Number of automatic retries on transient errors |

```python
from openai import AsyncOpenAI

# 1. Initialize async client
client = AsyncOpenAI()
```

---

### `class openai.AzureOpenAI(azure_endpoint, api_version, api_key)`

Client for models hosted on Azure OpenAI Service.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `azure_endpoint` | `str` | *Required* | Your Azure resource endpoint URL |
| `api_version` | `str` | *Required* | Azure API version string (e.g. `"2025-01-01"`) |
| `api_key` | `str` | *Required* | Your Azure OpenAI API key |

```python
from openai import AzureOpenAI

# 1. Initialize Azure client
client = AzureOpenAI(
    azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
    api_version="2025-01-01",
    api_key="..."
)
```

---

### `def client.with_options(timeout, max_retries)`

Returns a modified copy of the client without altering the original instance.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `timeout` | `float` | `None` | Override the default timeout for all requests on the new client |
| `max_retries` | `int` | `None` | Override the retry count for the new client |

```python
# 1. Create a scoped client variant for a low-latency call
client2 = client.with_options(timeout=5.0, max_retries=0)
```

---

## Chat Completions

### `def client.chat.completions.create(model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, n, logprobs, top_logprobs, stop, stream, seed, tools, tool_choice, response_format)`

Sends a messages array to a chat model and returns a completion response.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use (e.g. `"gpt-4o"`) |
| `messages` | `list[dict]` | *Required* | Conversation history as an array of role/content objects |
| `temperature` | `float` | `1.0` | Randomness control — lower is focused, higher is creative (range 0–2) |
| `max_tokens` | `int` | `None` | Hard cap on output tokens generated |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold — alternative to `temperature` |
| `frequency_penalty` | `float` | `0.0` | Reduces repetition of tokens already in the output (range 0–2) |
| `presence_penalty` | `float` | `0.0` | Encourages the model to introduce new topics (range 0–2) |
| `n` | `int` | `1` | Number of independent completion choices to return |
| `logprobs` | `bool` | `False` | Include log probabilities for output tokens |
| `top_logprobs` | `int` | `None` | Number of top-token log probs to return per position (requires `logprobs=True`) |
| `stop` | `str \| list[str]` | `None` | Stop generation when any of these sequences are produced |
| `stream` | `bool` | `False` | Enable raw delta-chunk streaming mode |
| `seed` | `int` | `None` | Fixed seed for deterministic sampling |
| `tools` | `list[dict]` | `None` | Function definitions the model can invoke |
| `tool_choice` | `str \| dict` | `"auto"` | Controls if/which tool the model calls |
| `response_format` | `dict` | `None` | Force JSON output (`{"type": "json_object"}`) or a specific schema |

```python
from openai import OpenAI

client = OpenAI()

# 1. Basic chat completion
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
)

# 2. Extract the response text
print(completion.choices[0].message.content)

# 3. With generation parameters
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku."}],
    temperature=0.7,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# 4. Return multiple choices
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Give me a name."}],
    n=3
)

# 5. Include log probabilities
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    logprobs=True,
    top_logprobs=5
)

# 6. Stop at specific sequences
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List items."}],
    stop=["END", "\n\n"]
)
```

---

## Structured Outputs

### `def client.chat.completions.parse(model, messages, response_format)`

Generates a response and automatically parses it into a typed Pydantic model.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use |
| `messages` | `list[dict]` | *Required* | Conversation history |
| `response_format` | `type[BaseModel]` | *Required* | A Pydantic model class defining the expected output shape |

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# 1. Define a Pydantic schema for the response
class Reply(BaseModel):
    answer: str
    confidence: float

# 2. Parse the response directly into the model
result = client.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    response_format=Reply
)

# 3. Access typed fields directly
print(result.choices[0].message.parsed.answer)
print(result.choices[0].message.parsed.confidence)

# 4. Alternative — force raw JSON output and parse yourself
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Return a JSON object."}],
    response_format={"type": "json_object"}
)
```

---

## Responses API

### `def client.responses.create(model, input, instructions)`

Primary Responses API — generates a response from a string or message array.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use (e.g. `"gpt-4.1"`) |
| `input` | `str \| list[dict]` | *Required* | Plain string prompt or a structured message array |
| `instructions` | `str` | `None` | System-level instruction without requiring an explicit system message |

```python
from openai import OpenAI

client = OpenAI()

# 1. Simple text response
response = client.responses.create(
    model="gpt-4.1",
    input="Explain recursion."
)
print(response.output_text)

# 2. With system instructions and a message array
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "Summarize this."}],
    instructions="You are a concise summarizer."
)
```

---

### `def client.responses.retrieve(response_id)`

Fetches a previously created and stored response by its ID.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `response_id` | `str` | *Required* | The ID of the stored response to retrieve |

```python
# 1. Fetch a stored response
response = client.responses.retrieve("resp_abc123")
```

---

### `def client.responses.delete(response_id)`

Permanently deletes a stored response.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `response_id` | `str` | *Required* | The ID of the stored response to delete |

```python
# 1. Delete a stored response
client.responses.delete("resp_abc123")
```

---

### `def client.responses.list()`

Lists all responses for your organization, paginated.

```python
# 1. Iterate over all stored responses
page = client.responses.list()
for r in page:
    print(r.id)
```

---

### `def client.responses.input_items.list(response_id)`

Lists the exact input items submitted with a specific response.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `response_id` | `str` | *Required* | The ID of the response whose input items to list |

```python
# 1. List input items for a response
items = client.responses.input_items.list("resp_abc123")
```

---

## Streaming

### `def client.chat.completions.stream(model, messages)`

Managed streaming context manager for chat completions — yields typed events as tokens arrive.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use |
| `messages` | `list[dict]` | *Required* | Conversation history |

```python
from openai import OpenAI

client = OpenAI()

# 1. Managed streaming context — yields typed chunks
with client.chat.completions.stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story."}]
) as stream:
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

# 2. Get the fully assembled completion after streaming ends
with client.chat.completions.stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
) as stream:
    final = stream.get_final_completion()

# 3. Raw streaming via create() — no context manager required
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Count to 10."}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

### `def client.responses.stream(model, input)`

Managed streaming context manager for the Responses API — yields server-sent events.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use |
| `input` | `str \| list[dict]` | *Required* | Plain string or message array |

```python
from openai import OpenAI

client = OpenAI()

# 1. Stream a Responses API call
with client.responses.stream(
    model="gpt-4.1",
    input="Write a poem."
) as stream:
    for event in stream:
        print(event)
```

---

## Async Usage

### `async def client.chat.completions.create(model, messages)`

Async chat completion — non-blocking, must run inside an event loop.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use |
| `messages` | `list[dict]` | *Required* | Conversation history |

```python
import asyncio
from openai import AsyncOpenAI

# 1. Initialize async client
client = AsyncOpenAI()

# 2. Define an async function for chat completion
async def main():
    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(completion.choices[0].message.content)

# 3. Run the async entry point
asyncio.run(main())

# 4. Async streaming example
async def stream_async():
    async with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Count to 5."}]
    ) as stream:
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")

asyncio.run(stream_async())
```

---

## Tool / Function Calling

### `def client.chat.completions.create(model, messages, tools, tool_choice)`

Extends chat completions with callable function definitions the model can invoke.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Model ID to use |
| `messages` | `list[dict]` | *Required* | Conversation history |
| `tools` | `list[dict]` | *Required* | One or more function definitions the model may call |
| `tool_choice` | `str \| dict` | `"auto"` | `"auto"` lets the model decide; `"none"` disables tools; a dict forces a specific function |

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. Define one or more callable tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

# 2. Let the model decide whether to call a tool
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# 3. Force a specific tool call
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather update please."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)

# 4. Extract the tool call name and arguments
tool_call = completion.choices[0].message.tool_calls[0]
print(tool_call.function.name)
print(json.loads(tool_call.function.arguments))

# 5. Send the tool result back to continue generation
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
messages.append(completion.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": '{"temp": "15C", "condition": "Cloudy"}'
})

follow_up = client.chat.completions.create(model="gpt-4o", messages=messages)
print(follow_up.choices[0].message.content)
```

---

## Vision (Image Input)

### `def client.chat.completions.create(model, messages)` — with image content

Accepts multimodal message content including text and image blocks in the same request.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | A vision-capable model ID (e.g. `"gpt-4o"`) |
| `messages[].content` | `list[dict]` | *Required* | Array of content blocks — each is `"text"` or `"image_url"` typed |
| `image_url.url` | `str` | *Required* | Public image URL or a base64 data URI (`data:image/png;base64,...`) |
| `image_url.detail` | `str` | `"auto"` | Resolution hint — `"low"`, `"high"`, or `"auto"` (affects token cost) |

```python
import base64
from openai import OpenAI

client = OpenAI()

# 1. Send a public image URL
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
            ]
        }
    ]
)
print(completion.choices[0].message.content)

# 2. Send a local image encoded as base64
with open("image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high"
                }}
            ]
        }
    ]
)
print(completion.choices[0].message.content)
```

---

## Embeddings

### `def client.embeddings.create(input, model, dimensions)`

Generates vector embeddings for a single string or a batch of strings.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `input` | `str \| list[str]` | *Required* | Text string or list of strings to embed |
| `model` | `str` | *Required* | Embedding model ID (e.g. `"text-embedding-3-small"`) |
| `dimensions` | `int` | `None` | Truncate the output vector length — reduces storage and compute cost |

```python
from openai import OpenAI

client = OpenAI()

# 1. Single string embedding
response = client.embeddings.create(
    input="The quick brown fox",
    model="text-embedding-3-small"
)
vector = response.data[0].embedding
print(len(vector))

# 2. Batch embedding for multiple strings
response = client.embeddings.create(
    input=["sentence one", "sentence two", "sentence three"],
    model="text-embedding-3-small"
)
vectors = [d.embedding for d in response.data]

# 3. Reduce output dimensions for storage efficiency
response = client.embeddings.create(
    input="Hello",
    model="text-embedding-3-large",
    dimensions=512
)

# 4. Check token usage
print(response.usage.total_tokens)
```

---

## Images (DALL-E)

### `def client.images.generate(prompt, model, n, size, response_format, quality, style)`

Generates one or more images from a text prompt using DALL-E.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `prompt` | `str` | *Required* | Text description of the image to generate |
| `model` | `str` | `"dall-e-2"` | Model ID — `"dall-e-2"` or `"dall-e-3"` |
| `n` | `int` | `1` | Number of images to generate |
| `size` | `str` | `"1024x1024"` | Output resolution (e.g. `"512x512"`, `"1024x1024"`) |
| `response_format` | `str` | `"url"` | `"url"` returns a temporary hosted link; `"b64_json"` returns raw base64 |
| `quality` | `str` | `"standard"` | `"hd"` for higher detail — DALL-E 3 only |
| `style` | `str` | `"vivid"` | `"vivid"` or `"natural"` rendering style — DALL-E 3 only |

```python
from openai import OpenAI

client = OpenAI()

# 1. Generate an image from a prompt
image = client.images.generate(
    prompt="A red fox sitting on a snowy mountain",
    model="dall-e-3",
    n=1,
    size="1024x1024"
)
print(image.data[0].url)

# 2. Return image as base64 instead of URL
image = client.images.generate(
    prompt="A futuristic city",
    model="dall-e-3",
    response_format="b64_json"
)

# 3. Edit an image by filling in masked (transparent) areas
image = client.images.edit(
    image=open("original.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="Add a sunset in the background",
    n=1,
    size="1024x1024"
)

# 4. Generate variations of an existing image
image = client.images.create_variation(
    image=open("original.png", "rb"),
    n=2,
    size="1024x1024"
)
print(image.data[0].url)
```

---

### `def client.images.edit(image, mask, prompt, n, size)`

Modifies an existing image by filling transparent mask regions guided by a prompt.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `image` | `BinaryIO` | *Required* | Source image file (PNG, RGBA supported) |
| `mask` | `BinaryIO` | *Required* | Mask image — transparent areas define the region to edit |
| `prompt` | `str` | *Required* | Description of what to generate in the masked region |
| `n` | `int` | `1` | Number of edited images to generate |
| `size` | `str` | `"1024x1024"` | Output resolution |

---

### `def client.images.create_variation(image, n, size)`

Generates `n` variations of an existing image without a text prompt.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `image` | `BinaryIO` | *Required* | Source image to vary |
| `n` | `int` | `1` | Number of variations to return |
| `size` | `str` | `"1024x1024"` | Output resolution |

---

## Audio

### `def client.audio.speech.create(model, voice, input)`

Converts text to spoken audio and returns streamable binary audio.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | TTS model ID — `"tts-1"` or `"tts-1-hd"` |
| `voice` | `str` | *Required* | Speaker voice — `"alloy"`, `"echo"`, `"fable"`, `"onyx"`, `"nova"`, or `"shimmer"` |
| `input` | `str` | *Required* | The text to synthesize into speech |

```python
from openai import OpenAI

client = OpenAI()

# 1. Text-to-speech — save to file
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, this is a text to speech test."
)
response.stream_to_file("output.mp3")

# 2. Streaming TTS — avoids buffering large audio files
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="nova",
    input="Streaming audio output."
) as response:
    response.stream_to_file("output.mp3")
```

---

### `def client.audio.transcriptions.create(file, model, response_format, timestamp_granularities)`

Transcribes an audio file to text using the Whisper model.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file` | `BinaryIO` | *Required* | Audio file object to transcribe |
| `model` | `str` | *Required* | Transcription model — e.g. `"whisper-1"` |
| `response_format` | `str` | `"json"` | Output format — `"json"`, `"text"`, `"srt"`, `"verbose_json"`, `"vtt"` |
| `timestamp_granularities` | `list[str]` | `None` | Include per-`"word"` or per-`"segment"` timestamps |

```python
from openai import OpenAI

client = OpenAI()

# 1. Transcribe audio to text
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )
print(transcript.text)

# 2. Transcription with word-level timestamps
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
```

---

### `def client.audio.translations.create(file, model)`

Transcribes audio in any language and translates the output to English.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file` | `BinaryIO` | *Required* | Audio file to translate |
| `model` | `str` | *Required* | Model ID — e.g. `"whisper-1"` |

```python
from openai import OpenAI

client = OpenAI()

# 1. Translate foreign audio to English
with open("audio.mp3", "rb") as f:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=f
    )
print(translation.text)
```

---

## Files

### `def client.files.create(file, purpose)`

Uploads a file for use with fine-tuning, batch, or assistants APIs.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file` | `BinaryIO` | *Required* | File object to upload |
| `purpose` | `str` | *Required* | Intended use — `"fine-tune"`, `"batch"`, `"assistants"`, etc. |

```python
from openai import OpenAI

client = OpenAI()

# 1. Upload a fine-tuning dataset
with open("training.jsonl", "rb") as f:
    file = client.files.create(file=f, purpose="fine-tune")
print(file.id)

# 2. Check file status
file = client.files.retrieve("file-abc123")
print(file.status)

# 3. Download raw file content
content = client.files.content("file-abc123")
print(content.text)

# 4. Delete a file
client.files.delete("file-abc123")

# 5. List all uploaded files
files = client.files.list()
for f in files:
    print(f.id, f.filename)

# 6. Block until server-side processing completes
file = client.files.wait_for_processing("file-abc123")
```

---

### `def client.files.retrieve(file_id)`

Returns metadata for an uploaded file including `filename`, `size`, `status`, and `created_at`.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file_id` | `str` | *Required* | ID of the file to retrieve |

---

### `def client.files.content(file_id)`

Downloads the raw bytes of an uploaded file.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file_id` | `str` | *Required* | ID of the file to download |

---

### `def client.files.delete(file_id)`

Permanently removes an uploaded file from your organization.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file_id` | `str` | *Required* | ID of the file to delete |

---

### `def client.files.wait_for_processing(file_id)`

Blocks and polls until server-side file processing completes.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `file_id` | `str` | *Required* | ID of the file to wait on |

---

## Fine-tuning

### `def client.fine_tuning.jobs.create(training_file, model)`

Starts a fine-tuning job on a base model using an uploaded JSONL training file.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `training_file` | `str` | *Required* | File ID of the uploaded JSONL training dataset |
| `model` | `str` | *Required* | Base model to fine-tune (e.g. `"gpt-4o-mini"`) |

```python
from openai import OpenAI

client = OpenAI()

# 1. Start a fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model="gpt-4o-mini"
)
print(job.id)

# 2. Check job status
job = client.fine_tuning.jobs.retrieve("ftjob-abc123")
print(job.status)  # queued | running | succeeded | failed

# 3. Cancel a running job
client.fine_tuning.jobs.cancel("ftjob-abc123")

# 4. List recent jobs
jobs = client.fine_tuning.jobs.list(limit=10)
for j in jobs:
    print(j.id, j.status)

# 5. Stream training log events
events = client.fine_tuning.jobs.list_events("ftjob-abc123")
for e in events:
    print(e.message)

# 6. List intermediate checkpoints
checkpoints = client.fine_tuning.jobs.checkpoints.list("ftjob-abc123")
for c in checkpoints:
    print(c.fine_tuned_model_checkpoint)
```

---

### `def client.fine_tuning.jobs.retrieve(job_id)`

Returns the status and details of a fine-tuning job.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `job_id` | `str` | *Required* | Fine-tuning job ID (e.g. `"ftjob-abc123"`) |

---

### `def client.fine_tuning.jobs.cancel(job_id)`

Aborts a queued or running fine-tuning job.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `job_id` | `str` | *Required* | Fine-tuning job ID to cancel |

---

### `def client.fine_tuning.jobs.list_events(job_id)`

Streams training log events including loss metrics at each step.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `job_id` | `str` | *Required* | Fine-tuning job ID to stream events from |

---

### `def client.fine_tuning.jobs.checkpoints.list(job_id)`

Lists saved mid-training checkpoints — each has a usable model ID.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `job_id` | `str` | *Required* | Fine-tuning job ID whose checkpoints to list |

---

## Batch

### `def client.batches.create(input_file_id, endpoint, completion_window)`

Submits a JSONL file of requests for async processing at 50% lower cost.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `input_file_id` | `str` | *Required* | File ID of the JSONL batch input |
| `endpoint` | `str` | *Required* | API endpoint for all requests (e.g. `"/v1/chat/completions"`) |
| `completion_window` | `str` | *Required* | Maximum processing window — currently only `"24h"` is supported |

```python
from openai import OpenAI

client = OpenAI()

# 1. Submit a batch of requests
batch = client.batches.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(batch.id)

# 2. Poll the batch status
batch = client.batches.retrieve("batch_abc123")
print(batch.status)  # validating | in_progress | completed | failed

# 3. Cancel before completion
client.batches.cancel("batch_abc123")

# 4. List all batch jobs
batches = client.batches.list()
for b in batches:
    print(b.id, b.status)
```

---

### `def client.batches.retrieve(batch_id)`

Polls the status of a batch job.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `batch_id` | `str` | *Required* | Batch job ID to check |

---

### `def client.batches.cancel(batch_id)`

Cancels a batch job before it finishes processing.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `batch_id` | `str` | *Required* | Batch job ID to cancel |

---

## Moderations

### `def client.moderations.create(input, model)`

Checks text or image content against OpenAI's usage policies — free to use.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `input` | `str \| list[dict]` | *Required* | Text string, or an array of typed content blocks (text + image) |
| `model` | `str` | *Required* | Moderation model ID (e.g. `"omni-moderation-latest"`) |

```python
from openai import OpenAI

client = OpenAI()

# 1. Check text for policy violations
result = client.moderations.create(
    input="Some text to check.",
    model="omni-moderation-latest"
)
print(result.results[0].flagged)      # True if any category was violated
print(result.results[0].categories)   # Per-category boolean flags
print(result.results[0].category_scores)  # Float confidence scores (0–1)

# 2. Moderate text and image together in one call
result = client.moderations.create(
    input=[
        {"type": "text", "text": "Check this text."},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
    ],
    model="omni-moderation-latest"
)
```

---

## Models

### `def client.models.list()`

Lists all models your API key can access, including fine-tuned models.

```python
from openai import OpenAI

client = OpenAI()

# 1. List available models
models = client.models.list()
for m in models:
    print(m.id)
```

---

### `def client.models.retrieve(model_id)`

Returns metadata for a model — `id`, `created`, `owned_by`.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model_id` | `str` | *Required* | Model ID to retrieve (e.g. `"gpt-4o"`) |

```python
# 1. Get details about a specific model
model = client.models.retrieve("gpt-4o")
print(model.created, model.owned_by)
```

---

### `def client.models.delete(model_id)`

Permanently deletes a fine-tuned model that belongs to your organization.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model_id` | `str` | *Required* | Fine-tuned model ID to delete |

```python
# 1. Delete a fine-tuned model
client.models.delete("ft:gpt-4o-mini:org:custom:id")
```

---

## Error Handling

### Exception Classes

The SDK raises typed exceptions for every failure category — catch them individually or fall back to the base `APIStatusError`.

| Exception | HTTP Status | Description |
|---|---|---|
| `openai.AuthenticationError` | 401 | Invalid, missing, or revoked API key |
| `openai.RateLimitError` | 429 | Too many requests — implement exponential backoff |
| `openai.BadRequestError` | 400 | Malformed parameters or invalid input content |
| `openai.APIConnectionError` | — | SDK cannot reach the API — network or DNS failure |
| `openai.APIStatusError` | any non-2xx | Base class exposing `.status_code` and `.response` |

```python
import time
import openai
from openai import OpenAI

client = OpenAI()

# 1. Catch all major error types
try:
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
except openai.AuthenticationError as e:
    print("Invalid API key:", e.status_code)
except openai.RateLimitError as e:
    print("Rate limited — back off:", e.status_code)
except openai.BadRequestError as e:
    print("Bad request:", e.body)
except openai.APIConnectionError:
    print("Network error — check connection")
except openai.APIStatusError as e:
    print(f"HTTP {e.status_code}: {e.response}")

# 2. Simple exponential backoff on rate limit
for attempt in range(3):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        break
    except openai.RateLimitError:
        time.sleep(2 ** attempt)
```

---

## Pagination

### Auto-Pagination

All `.list()` methods support transparent auto-pagination — the SDK fetches subsequent pages automatically when you iterate.

```python
from openai import OpenAI

client = OpenAI()

# 1. Auto-paginate — SDK fetches next pages automatically
for model in client.models.list():
    print(model.id)

# 2. Manual cursor-based pagination
cursor = None
while True:
    page = client.fine_tuning.jobs.list(after=cursor, limit=20)
    for job in page.data:
        print(job.id)
    if not page.has_next_page():
        break
    cursor = page.data[-1].id
```

| Attribute / Method | Description |
|---|---|
| `page.data` | List of resource objects on the current page |
| `page.has_next_page()` | Returns `True` if more pages are available |
| `page.next_page()` | Manually fetches and returns the next page object |
| `after` param | Cursor for manual pagination — pass the last item's ID |

---

## Realtime API (WebSocket)

### `async def client.beta.realtime.connect(model)`

Opens a persistent WebSocket session for real-time audio or text exchange.

| Parameter | Type | Default | Description |
|---|---|:---:|---|
| `model` | `str` | *Required* | Realtime model ID (e.g. `"gpt-4o-realtime-preview"`) |

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    # 1. Open a persistent WebSocket session
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:

        # 2. Configure session modalities
        await conn.session.update(session={"modalities": ["text"]})

        # 3. Inject a user message into the live context
        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello!"}]
        })

        # 4. Prompt the model to respond
        await conn.response.create()

        # 5. Iterate over server-sent events
        async for event in conn:
            if event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                break

asyncio.run(main())
```

| Method | Description |
|---|---|
| `conn.session.update(session=...)` | Reconfigure modalities, voice, tools, or turn detection mid-call |
| `conn.conversation.item.create(item=...)` | Inject a message or function result into the live conversation |
| `conn.response.create()` | Prompt the model to generate a response on the current context |
| `async for event in conn` | Iterate over events — text deltas, audio bytes, function calls |
| `event.type` | Route events by type: `"response.text.delta"`, `"response.done"`, etc. |

---

## Utilities

### Response Utilities

Helper attributes and methods available on every SDK response object.

| Attribute / Method | Type | Description |
|---|---|---|
| `response._request_id` | `str` | Server-assigned `x-request-id` — include in bug reports or support tickets |
| `response.model_dump()` | `dict` | Serialize any Pydantic response object to a plain Python dictionary |
| `response.to_json()` | `str` | Serialize a response object to a JSON string |
| `completion.system_fingerprint` | `str` | Backend config identifier — changes signal non-determinism even with a fixed seed |
| `completion.usage.prompt_tokens` | `int` | Tokens consumed by the input messages |
| `completion.usage.completion_tokens` | `int` | Tokens generated in the model's response |
| `completion.usage.total_tokens` | `int` | Sum of `prompt_tokens` and `completion_tokens` |

```python
from openai import OpenAI

client = OpenAI()

# 1. Make a completion with a fixed seed for reproducibility
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Pick a number 1-10."}],
    seed=42
)

# 2. Inspect the server-assigned request ID for support tickets
print(completion._request_id)

# 3. Check the backend fingerprint for determinism verification
print(completion.system_fingerprint)

# 4. Inspect token usage
print(completion.usage.prompt_tokens)
print(completion.usage.completion_tokens)
print(completion.usage.total_tokens)

# 5. Serialize the response object
data = completion.model_dump()     # -> dict
json_str = completion.to_json()    # -> str

# 6. Use NOT_GIVEN to explicitly omit an optional parameter
import openai
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stop=openai.NOT_GIVEN   # distinct from None — omits the field entirely
)
```