# UphillConf Documents

## Why Open Source

### Collaboration and Innovation

- **Pooling Resources**: Collaborate to fix bugs, improve quickly, and build on each other's work.
- **Fostering Innovation**: Encourage new discoveries, bias detection, and efficiency improvements through shared access to code and data.

### Inclusion and Accessibility

- **Broad Access**: Diverse groups can access and contribute to open-source projects, enhancing inclusivity.
- **Educational Value**: Students and non-tech users can learn AI and other technologies through open resources.

### Speed and Flexibility

- **Rapid Development**: Diverse input accelerates progress.
- **IT Flexibility**: Users can train and deploy models anywhere, with the freedom to switch models as needed.

### Transparency and Security

- **Transparency**: Open access to models and data ensures visibility, accountability, and recourse.
- **Security**: Open source enhances defense against malicious activity and reduces single points of failure.

### Economic and Ecosystem Benefits

- **Economic Opportunity**: Supports a diverse market of complementary products.
- **Ecosystem Development**: Builds communities around open models, fostering collaborative growth.

### Open models vs. Closed APIs

While ML researchers and scientists inherently appreciate the benefits of open models for their reproducibility and deeper investigative potential, software engineers building products on top of LLMs should also consider open models over closed APIs. Firstly, open models enhance product reliability by eliminating the single point-of-failure associated with closed APIs, allowing the use of multiple API providers and libraries that optimize availability and latency. Secondly, open models facilitate a seamless transition to local deployment, reducing costs and latency without altering prompts or logic, thus minimizing technical debt. Lastly, they offer greater flexibility in balancing latency and cost, supporting diverse infrastructure needs, and enabling tailored solutions for different user segments and product features.

## Software Engineering Engagement is Critical

### Practical Integration of AI in Everyday Products

Software engineers (SWEs) are the builders of today’s digital products. To see how AI can enhance the tools and services we use daily, it is essential to empower SWEs to integrate AI technologies into these offerings. Their expertise is crucial in implementing AI in a way that is both effective and user-friendly.

### Diversity

Diversity in software engineering teams is vital for AI development. Diverse teams bring a variety of perspectives, which leads to more innovative solutions and reduces the risk of biases in AI systems. This diversity ensures that AI technologies are developed in a way that is inclusive and considers the needs of a broad user base.

### Multi-Disciplinary Collaboration

SWEs occupy a unique position within multi-disciplinary teams, working alongside product specialists, business leaders, and design professionals. They often serve as key advisors on feasibility and help set realistic expectations for new ideas. AI has transformed the conversation around what is possible, enabling greater creativity, particularly from Product Designers. However, understanding the current capabilities and limitations of AI is crucial to avoid costly mistakes and ensure that experimentation stays within realistic bounds.

### Ethical Guidance

Software engineers play a critical role in guiding the ethical use of AI, similar to their role in ensuring accessibility, security, and privacy in traditional software development. They can help navigate the complex ethical landscape of AI, ensuring that its deployment is responsible and aligns with societal values.

By engaging software engineers in AI development, we not only leverage their technical expertise but also ensure that AI is integrated thoughtfully, ethically, and inclusively into our everyday products and services.

## Understanding AI Capabilities

One of the challenges with incorporating machine leaning into an application is understanding the capabilities and limitations of machine learning. While there is no silver bullet, as the field is dynamic and changes frequently, there are a few principle that we can follow in order to broadly assess what machine learning can do and whether to not it will work for our use case.

### Prediction engines

At their core, machine learning models function as sophisticated prediction engines. These models identify patterns within data and make informed predictions based on those patterns. Whether predicting the next word in a sentence or forecasting stock market trends, the goal is to use historical data to predict future outcomes.

The process starts with training the model on a dataset where the outcomes are known. By comparing its predictions with actual outcomes, the model adjusts its parameters to improve accuracy. This training allows the model to generalize from specific examples to broader applications, making it capable of handling new data.

In essence, the predictive nature of machine learning models makes them powerful tools for anticipating future events and trends but also highlight their limitations. The quality and breadth of the data that is used for training often dictates how well-suited a model will be for a task.

### Understanding modalities

Machine learning modalities are the various types of data that machine learning models can process, such as text, images, audio, and video. Each modality requires specialized techniques and architectures, but more importantly give strong indications on whether or not they will be suitable for a given task or set of data. By handling different modalities, machine learning models can be applied across diverse applications, from speech recognition to image classification and beyond.

Common modalities include images, audio, video, text. Some models support multiple modalities.

### Understanding tasks

Machine learning tasks are less clearly defined but generally refer to a specific goal that a user may have in mind. The task is often dependent on usage because some models possess a variety of capabilities, some of which only present themselves when used in a specific way.

An example of this is a Large Language Model. These models possess a variety of capabilities and are suitable for a variety of tasks, such as summarisation and classification.

So while the modality or modalities of a model can be clearly defined, the tasks that a model is capable of often requires some experimentation. Some models are designed with a very specific tasks in mind but others are more general in their capabilities. Often, creative use of such models can result in new and interesting tasks that had not been previously considered.

Examples of tasks include classification, regression, translation, and generation.

### Modality-task matrix

When selecting a model both the data you have available and the thing you wish to accomplish have to be considered at all times. With this in mind I have developed a simple matrix that can help you to asses which model might work for you.

This matrix is definitely a simplification but offers a good starting point. The modality of a model should be matched to the data you have available to input and the tasks should align (roughly) with your end goal.

| Modality    | Classification          | Regression                   | Clustering            | Generation       | Translation          | Detection                | Segmentation       |
| ----------- | ----------------------- | ---------------------------- | --------------------- | ---------------- | -------------------- | ------------------------ | ------------------ |
| Text        | Sentiment Analysis      | Sentiment Score Prediction   | Topic Clustering      | Text Generation  | Language Translation | Named Entity Recognition | -                  |
| Images      | Image Classification    | Image Quality Estimation     | Image Clustering      | Image Synthesis  | Image Captioning     | Object Detection         | Image Segmentation |
| Audio       | Speaker Identification  | Emotion Intensity Prediction | Sound Clustering      | Audio Generation | Speech-to-Text       | Sound Event Detection    | -                  |
| Video       | Activity Recognition    | -                            | Scene Clustering      | Video Generation | -                    | Action Detection         | Video Segmentation |
| Tabular     | Customer Classification | Sales Prediction             | Customer Segmentation | -                | -                    | Anomaly Detection        | -                  |
| Time Series | Fault Detection         | Stock Price Prediction       | Pattern Clustering    | -                | -                    | Anomaly Detection        | -                  |
| Graphs      | Node Classification     | -                            | Community Detection   | Graph Generation | -                    | Link Prediction          | -                  |

Using this matrix you may (no guarantees) be able to use more specific language to filter out more models leaving fewer candidates to assess.

### Finding a model

Once you know what kind of model you want to use and have a certain degree of confidence that it could theoretically exist. You are left with the sometimes intimidating task of finding one.

Luckily, there are a few options available to us.

#### Tasks

The simplest approach is to the Hugging Face Tasks page to explain in more detail how to achieve a specific goal and what models might be appropriate.

[Hugging Face Tasks](https://huggingface.co/tasks)

This section of the Hugging Face site can help point you in the right direction, although it isn’t exhaustive, but more importantly it will offer some examples of how execute the specific task, while explaining some of the challenges and limitations. If the exact task you are looking for does not exist, then it may be beneficial to select the most similar task and explore the subject from there.

#### Hugging Face Model Hub

Another way to find models is directly on the Hugging Face Hub, making use of the ability to search and filter by a variety of labels, tasks, and libraries.

[Hugging Face Models](https://huggingface.co/models)

Models are tagged with a variety of metadata allowing for some relatively fine filtering. While this won’t reduce the number of models down to one, it will significantly reduce the number of candidates.

## Gradio

Gradio is a python library that allows you to generate a user interface for a machine learning model (or for any reason really) very simple.

While the benefits to a software engineer might not be immediately apparent, it is still useful, even if you are comfortable building your own UIs.

### Benefits of gradio

#### Rapid Experimentation

Gradio accelerates the creation of user interfaces for ML models, enabling quick prototyping and testing with or without extensive JavaScript and HTML knowledge. Its intuitive API allows developers to build interactive interfaces with minimal code, facilitating immediate feedback and iterative model improvement. Gradio also offers flexibility for customizing interfaces to adapt experiments based on initial results.

#### Testing and Benchmarking

Gradio simplifies testing and benchmarking by providing consistent interfaces for model evaluation, ensuring fair comparisons. It facilitates easy sharing of interfaces, enabling collaborative testing and peer reviews through simple links. Additionally, Gradio can integrate into automated testing pipelines, supporting continuous integration and delivery (CI/CD) practices in ML workflows.

#### Capitalizing on the Python Ecosystem

Gradio leverages Python's rich ML ecosystem, integrating seamlessly with popular libraries like TensorFlow, PyTorch, and Scikit-learn. The extensive Python community offers abundant resources and support. Gradio interfaces can be reused across projects, promoting efficient workflows and code reuse, especially in collaborative environments. This makes it especially beneficial, even to experienced frontend developers, as while the JavaScript AI ecosystem is gradually improving the python ecosystem offers a much larger number of interesting libraries and integrations, many of while require your to be as close to the models as possible (i.e. integrating with PyTorch et al), many of these python libraries have capabilities that are impossible or very difficult for other languages.

### Using gradio

There are two main approaches to building gradio demos.

Either using a high level API such as `Interface` or `ChatInterface` or the lower level `Blocks` API.

#### `Interface`

This API allows you to spin up a UI with just a few lines of python. While this approach is very simple and very fast you don’t have as much control over the UI.

```py
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

`ChatInterface` is similar to `Interface` except it is specifically designed for building chatbots.

Refer to the documentation for more info on [`gr.Interface`](https://www.gradio.app/guides/the-interface-class) and [`gr.ChatInterface`](https://www.gradio.app/guides/creating-a-chatbot-fast).

#### `Blocks`

The blocks API is a much more powerful and flexible API but is a little more complex with more things to consider.

At its core, it is simply a tree of UI components and some event listeners that dictate how the app should behave.

In gradio there are four important concepts when working with event listeners: the target, the inputs, the outputs, and the prediction function.

- **target**: The target is the component that triggers the event you care about. For example in `gr.Button().click(...)` the button is the target that is emitting the event. This is similar to DOM events on the web.
- **inputs**: The inputs are the the components whose values will be passed to the predict function, inputs in gradio usually refer to components and not arbitrary data like in other UI frameworks. For example if a Textbox is an ‘input’ for a specific event, then the value of that textbox will be extracted and passed to the predict function.
- **outputs**: The outputs are the components that will receive the data that is returned from. For example if you were generating an image from some text, then you would mark an Image component as the output for a specific event. When the prediction had complete, gradio would pass that image data to the Image component.
- **prediction function**: sometimes just referred to as ‘the function’ or an ‘event handler’, this is function that performs the inference. It received values from the input components, and the value returned from this function is passed to the output components.

Together this looks a little like this:

```py
import gradio as gr

def predict(text):
  prediction = make_image(text)
  return prediction

with gr.Blocks() as demo:
  text = gr.Textbox()
  image = gr.Image()
  btn = gr.Button(“Run Prediction”)

  btn.click(predict, inputs=text, outputs=image)

demo.launch()
```

In this example we are triggering the function `predict` when a specific `Button` is `click`ed. The `predict` function will receive the value of the `Textbox` which is always a string and its return value will be passed to the `Image` component, which always expects images data.

Different components provide different types of data when used as inputs and expect a certain type of data when used as outputs. There are many components in gradio and these details along with other information can be found in the [component documentation](https://www.gradio.app/docs/gradio/introduction).

For more information on gradio blocks, [the guide](https://www.gradio.app/guides/blocks-and-event-listeners) offers a great place to start.

## Using machine learning models

There are many ways to use a machine learning models, we will look at a few examples.

### Hugging Face Inference API

The Hugging Face Inference API allows you to use a variety of different models for free via an API.

The supported models change over time but typically include some of the most important or popular models at any given time. For example, you can currently access a variety of Meta’s Llama 3 models and certain Mistral models via the Inference API.

See the [Inference API docs](https://huggingface.co/docs/api-inference/en/index) for more info.

#### Limits

In order for the Inference API to be sustainable there are certain limits, these limits are not currently documented but you generally have access to smaller models for free and have a request allowance that should suffice for experimentation.

To access large models (such as 70B parameter models) you can subscribe to Hugging Face Pro (around 10USD per month). This also increases your request allowance.

You can read more [here](https://huggingface.co/docs/api-inference/en/faq).

#### Using the Inference API

The inference API is accessible via either the JavaScript client or the Python client.

##### JavaScript

Install the JavaScript inference client

```bash
npm install @huggingface/inference
```

Create a client instance

```ts
import { HfInference } from '@huggingface/inference';

const hf = new HfInference(hf_access_token);
```

Make a request:

```ts
await hf.textGeneration({
	model: 'gpt2',
	inputs: 'The answer to the universe is'
});

for await (const output of hf.textGenerationStream({
	model: 'google/flan-t5-xxl',
	inputs: 'repeat "one two three four"',
	parameters: { max_new_tokens: 250 }
})) {
	console.log(output.token.text, output.generated_text);
}
```

For more information and to see how to perform other tasks, [check the documentation](https://huggingface.co/docs/huggingface.js/inference/README#natural-language-processing).

##### Python

Install the Python hub client library

```bash
pip install huggingface_hub
```

Create a client instance

```py
from huggingface_hub import InferenceClient
client = InferenceClient(token=hf_access_token)
```

Make a prediction:

```py
image = client.text_to_image(
  "An astronaut riding a horse on the moon."
)
image.save("astronaut.png")
```

Further information can be found on the [hub docs page](https://huggingface.co/docs/huggingface_hub/guides/inference#using-a-specific-model)

### `llama.cpp`

`LLama.cpp` is a library that allows you run a variety of models locally, on your machine. A variety of operating system are supported and there is usually a path to getting things working regardless of what CPU, GPU, or architecture you are running.

We will be focusing on `llama-cpp-python` but the core library is worth exploring too. You can find more information on the [GitHub repository for llama.cpp](https://github.com/ggerganov/llama.cpp).

#### Quantisation

##### What it is

Quantisation is a technique that reduces the precision of the numbers used to represent a model's parameters, such as weights and activations. This typically involves converting 32-bit floating-point numbers to lower-bit formats like 16-bit or 8-bit integers.

##### Why it matters

Quantisation is important because it can significantly reduce the memory footprint and computational requirements of machine learning models. This makes it possible to run complex models on resource-constrained devices, such as smartphones and embedded systems, without a substantial loss in accuracy.

##### How to find specific quants

Quantised versions of models are now readily available on the Hugging Face Hub. You can usually simply filter the results using the relevant library or tag. For LLama.cpp we are specifically interested in the GGUF format that is found in the ‘Libraries” tab. Many different quantisations are available, the Q4 quantisations typically perform well.

#### `llama-cpp-python`

`llama-cpp-python` offers a simple python interface for llama.cpp making integrating it into other python application very straightforward.

Installing can be complex but the installation instructions are relatively thorough, [find the relevant section for your hardware and follow the instructions](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation).

When we have everything installed usage is pretty simple. First we create an instance of the model:

```py
from llama_cpp import Llama

llm = Llama(
      model_path="./models/7B/llama-model.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
```

Then we can use it:

```py
output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)

print(output)
```

This library, like llama.cpp itself, is very powerful. I recommended [reading the docs](https://llama-cpp-python.readthedocs.io/en/stable/) for more specific information.

## Controlling LLMs

While LLMs are powerful tools getting the most out of them can be challenging, not least because of how you need to ‘guide’ them produce relevant and accurate output.

Here we list a few approaches to maximising the effectiveness of LLMS.

### Prompt Engineering

Prompt engineering is crucial for effectively using large language models. By crafting precise and well-structured prompts, you can significantly improve the model's performance and ensure that it produces relevant and accurate responses.

While prompt engineering sounds fancy, it isn’t an exact science and results will vary depending on the models being used. Prompt Engineering is valuable as a skill set but also a helpful framing technique. The real benefits comes from experimentation and being very intentional about the prompts that you craft.

#### Types

Prompt engineering can involve several techniques, including:

1. **Zero-Shot Prompting**: Asking the model to perform a task without providing examples.
2. **Few-Shot Prompting**: Giving a few examples within the prompt to guide the model.
3. **Chain-of-Thought Prompting**: Encouraging the model to explain its reasoning process step-by-step.
4. **Contextual Prompting**: Providing additional context or background information to improve the model’s responses.

#### Examples

- **Text Classification**: "Classify the text into neutral, negative, or positive. Text: 'This movie is great!' Sentiment: "
  - Here we are ‘pre-filling’ the response, so that the LLM can finish it off. Doing this increases the chance of getting a good response, because the number of likely different response for this start text is much smaller, than if it we left it open ended.
- **Named Entity Recognition**: "Extract the names of people, organizations, and places from the following text: 'John works at Hugging Face inNew York.'"
  - In this example, we clearly describe which ‘classes’ we are interested in, increasing the chance that we will get a meaningful response.
- **Translation**: "Translate the following sentence from English to French: 'Hello, how are you?' Translation: "
  - Again, we pre-fill the response text in the hopes of getting a more relevant result.
- **Content Generation**: "Write a short story about a dragon who wants to learn to cook"
  - This is a much more open ended prompt and great for more creative use cases such as content generation. Sometimes the unpredictable nature of LLMs is a huge benefit.

#### Resources

To learn more about prompt engineering, you can explore the following resources:

1. **Prompt Engineering Guide**: A comprehensive guide that covers various techniques and applications of prompt engineering [promptingguide.ai](https://www.promptingguide.ai).
2. **GitHub Prompt Engineering Guide**: Offers practical tips and examples for developers [GitHub](https://github.com/dair-ai/Prompt-Engineering-Guide).
3. **OpenAI Documentation**: Provides guidelines and best practices for using prompts with OpenAI models [OpenAI Platform](https://platform.openai.com/docs/guides/prompt-engineering).
4. **Hugging Face Prompting Guide**: Focuses on using Hugging Face models for various NLP tasks [Hugging Face](https://huggingface.co/docs/transformers/main/tasks/prompting).
5. **Anthropic Prompt Engineering Guide**: Detailed documentation on prompting approaches for Anthropic models [Anthropic](https://docs.anthropic.com/en/docs/prompt-engineering)
6. **Mistral Prompting Capabilities**: Focuses on suing Mistral models [Mistral](https://docs.mistral.ai/guides/prompting_capabilities/)

While some of these resources are model specific, they are all full of very useful information and can be a great starting point for prompting any model. In many cases the general approaches detailed in the model-specific guides can be followed with any model.

In essence, you want to have a variety of techniques and approaches at your disposal and start experimenting, refining until you get results you are happy with.

Assessing the output on ‘feel’ is a perfectly acceptable approach until you decide you want to benchmark and assess results more systematically.

### Gaining more control

While the above prompting techniques can help to get relevant and accurate responses, there is still plenty of opportunity for the model to hallucinate, generate very long output, or just generally do something undesirable, even if the output is generally correct.

An example of this is our first prompt example.

```
"Classify the text into neutral, negative, or positive. Text: 'This movie is great!' Sentiment: "
```

While the expectation here is pretty clear, the model could still produce output that isn’t quite what you are expecting. In an ideal scenario, the model oils produce just a single word (the most likely class), however it may ‘get creative’. It is perfectly reasonable for the LLM to produce any of the following outputs:

- “positive” - great
- “Positive” - capitalised but fine
- “Mostly positive” - this wasn’t a class
- “Positive. The language used is mostly speaking in favour of the subject” - The added explanation is not ideal

In cases where the output will be used in a ‘natural language’ context, i.e. a human is going to read it for refine it further, this isn’t a huge problem even if it isn’t ideal. But in cases where we want to do something programmatic with the output, it is entirely possible that any output other than the first ‘correct’ response could be completely useless, even though all of the responses are correct.

#### Guided output

In an ideal world we could leverage the excellent predictive skills of an LLM while getting guarantees about the shape of the output.

Thankfully this is possible with guided or constrained output. A number of libraries facilitate this by taking a schema of some description that describes the output and manipulating the LLMs prediction engine to guarantee that only outputs that precisely fit the schema are possible.

#### outlines

Outlines is a library that allows you to define a schema to guarantee a certain output. It has integrations with `llama-cpp-python`, `transformers`, and a variety of other libraries.

With outlines we can do the following, knowing exactly what output we are going to get back:

```py
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.choice(
  model,
  ["neutral", "negative", "positive"]
)

sentiment = generator("Classify the following text. Text: 'This movie is great!' Sentiment:  ")

print(sentiment)
```

##### JSON Schema

I particularly like outlines because it has a great API for definite a JSON, schema. JSON-based output is one of the more useful forms of constrained output from an LLM. It is essentially for function calling and great for when you want to programmatically to something with the output directly.

`outlines` using Pydantic types to define the schema. Pydantic is a data modelling and validation library in python and it makes definite data models very straightforward.

This is an example:

```py
from pydantic import BaseModel
import outlines


class Person(BaseModel):
    name: str
    age: int
    location: str

model = outlines.models.transformers(
  "mistralai/Mistral-7B-Instruct-v0.2"
)
generator = outlines.generate.json(
  model,
  Character
)

person = generator(
    "Extract the details from this text: "
    + "Frances is 27 and lives in New York City"
    )

print(person)

# name=“Frances”
# age=27
# location=“New York City”
```

This is a powerful way to use the full power of an LLM while still guaranteeing as certain output shape.

I’d encourage you to [check out the docs](https://outlines-dev.github.io/outlines/) for more information. Outlines is capable of many things.

### Other ways of using ML models

- **[`transformers`](https://huggingface.co/docs/transformers/index)**:train and run almost anything.
- **[`diffusers`](https://huggingface.co/docs/diffusers/index)**: For diffusion models such as many image and audio generation models
- **[`sentence-transformers`](https://sbert.net)**: for text and image embedding models
- **[`mlx`](https://ml-explore.github.io/mlx/build/html/index.html)**: run machine learning models on apple hardware
