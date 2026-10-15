from transformers import pipeline

classifier = pipeline(
    task = "sentiment-analysis",
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I absolutely love learning python!",
    "I loved this movie",
    "Worst experience ever",
    "it was okay, nothing special.",
    "i expected it to be terrible, but it was amazing!"
]

results = classifier(texts)

print(results)

for text, result in zip(texts,results):
    print(text)
    print(result)
    print()