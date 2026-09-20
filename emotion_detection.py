stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)
df = pd.read_csv('/content/tweet_emotions.csv.zip')

df['clean_text'] = df['content'].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Precision:", precision_score(
        y_test,
        y_pred,
        average='macro'
    ))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def predict_emotion(text):
    try:
        result = emotion_classifier(text)

        if (
            isinstance(result, list)
            and len(result) > 0
            and isinstance(result[0], list)
            and len(result[0]) > 0
            and 'label' in result[0][0]
        ):
            return result[0][0]['label']

        else:
            print("Unexpected result:", result)
            return "neutral"

    except Exception as e:
        print("Error:", e)
        return "neutral"

plt.figure(figsize=(8, 5))

sns.countplot(
    x='sentiment',
    hue='sentiment',
    data=df,
    palette='viridis',
    legend=False
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()