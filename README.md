# Personality Classification — Introvert vs Extrovert

A binary classification project that predicts whether a person is an
**Introvert** or **Extrovert** based on behavioral and social features
(time spent alone, social event attendance, stage fear, friends circle
size, posting frequency, etc.).

## Dataset

- 2,900 rows, 8 columns, no missing values
- Balanced target: ~51% Extrovert / ~49% Introvert
- Features: `Time_spent_Alone`, `Stage_fear`, `Social_event_attendance`,
  `Going_outside`, `Drained_after_socializing`, `Friends_circle_size`,
  `Post_frequency`
- Target: `Personality` (Introvert / Extrovert)

## Approach

1. Encode categorical columns (`Stage_fear`, `Drained_after_socializing`,
   `Personality`) with `LabelEncoder`.
2. Split into train/test (80/20, stratified on the target to preserve
   class balance).
3. Scale features with `StandardScaler` for the neural network.
4. Train and evaluate two models:
   - **Logistic Regression**
   - **A small feed-forward neural network** (3 hidden layers, ReLU,
     sigmoid output)
5. Evaluate both models with accuracy, precision/recall/F1
   (`classification_report`), and a confusion matrix — not accuracy
   alone, since it can be misleading even on balanced data.

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~0.92    |
| Neural Network       | ~0.90   |

Both classes (Introvert/Extrovert) get comparable precision and recall,
which confirms the model isn't just favoring one class.

## Business Insights

- **Time spent alone** and **feeling drained after socializing** are the
  strongest indicators of introversion.
- **Social event attendance**, **going outside**, and **posting
  frequency** are the strongest indicators of extroversion.
- Logistic Regression performs about as well as the neural network,
  suggesting the relationship between these features and personality
  type is close to linear — added model complexity isn't buying much
  extra accuracy here.
- **Potential use cases:** personalized content recommendations,
  onboarding flows, or team-composition tools that benefit from a
  quick personality signal instead of a full questionnaire.
- **Limitation:** this is a controlled/synthetic benchmark dataset, so
  real-world data (e.g. from actual social media behavior) would
  likely be noisier, and production accuracy would probably be lower.

## Files

- `Personality Classification — Introvert vs Extrovert.ipynb` — full notebook with markdown
  explanations, run top to bottom
- `personality_datasert.csv` — dataset (place in the same folder before
  running)

## How to run

```bash
pip install pandas scikit-learn tensorflow
jupyter notebook Personality Classification — Introvert vs Extrovert.ipynb
```

## Notes

- This is a learning/portfolio project. The dataset's features
  correlate strongly with the target, which is why accuracy is high —
  this is a well-known characteristic of this dataset, not a sign of
  data leakage (verified: no feature is a direct transformation of
  the label).
