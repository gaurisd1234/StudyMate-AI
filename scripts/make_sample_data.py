"""Create a small sample PDF + question set so you can test the pipeline right away.

Run:  python scripts/make_sample_data.py
Creates: data/sample/ml_notes.pdf  and  data/sample/qa.csv

These are generic machine-learning notes. Replace them with YOUR OWN material
(put PDFs in data/docs/ and write your questions in data/eval/qa.csv).
"""
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"

PAGES = [
    ("Linear Regression",
     "Linear regression models the relationship between one or more input features and a continuous target by fitting a straight line (or a hyperplane when there are several features). The model predicts y as a weighted sum of the inputs plus an intercept. Training means finding the weights that make the predictions as close as possible to the true values. The most common approach is ordinary least squares, which chooses the weights that minimize the sum of squared differences between predicted and actual values. The average of these squared differences is called the mean squared error (MSE), and its square root, the RMSE, is expressed in the same units as the target. Linear regression assumes a roughly linear relationship, errors that are independent, and features that are not strongly correlated with each other. When features are highly correlated, the learned weights become unstable, a problem known as multicollinearity. Residual plots help check whether the model's assumptions hold."),
    ("Logistic Regression and Classification Metrics",
     "Logistic regression is a classification model that predicts the probability that an example belongs to the positive class. It passes a weighted sum of the features through the sigmoid function, which squeezes any number into the range between zero and one. A threshold, usually 0.5, turns the probability into a class label. Classification models are judged with a confusion matrix, which counts true positives, false positives, true negatives and false negatives. Precision is the fraction of predicted positives that are truly positive, while recall is the fraction of real positives that the model finds. The F1 score is the harmonic mean of precision and recall, giving a single number that balances both. Plain accuracy can be misleading on imbalanced data: a model that always predicts the majority class can score ninety-nine percent accuracy and still be useless. In such cases, look at per-class recall, the F1 score, or the area under the ROC curve."),
    ("Decision Trees and Random Forests",
     "A decision tree classifies data by asking a sequence of yes-or-no questions about the features. At each node, the algorithm chooses the split that makes the resulting groups as pure as possible, measured by Gini impurity or entropy. A single deep tree can memorize the training data and overfit. A random forest reduces this problem by training many trees, each on a random bootstrap sample of the data and using only a random subset of features at each split. This technique of averaging many different models is called bagging. The final prediction is the majority vote of the trees for classification, or the average for regression. Because the individual trees make different mistakes, their errors partly cancel out, which lowers the variance of the ensemble. Random forests also provide feature importance scores, though these can be biased toward features with many distinct values."),
    ("Convolutional Neural Networks",
     "A convolutional neural network, or CNN, is designed for data with a grid structure such as images. Instead of connecting every pixel to every neuron, a CNN slides small filters, also called kernels, across the image. Each filter detects a local pattern like an edge or a texture and produces a feature map. The same filter weights are reused at every position, an idea called weight sharing, which greatly reduces the number of parameters and lets the network detect a pattern wherever it appears. Pooling layers, such as max pooling, shrink the feature maps by keeping only the strongest response in each small region, which adds some tolerance to small shifts and reduces computation. Early layers learn simple features and deeper layers combine them into complex shapes. Transfer learning reuses a CNN pretrained on a large dataset such as ImageNet and fine-tunes it on a smaller task."),
    ("Recurrent Networks and LSTM",
     "Recurrent neural networks process sequences such as text or time series one step at a time, carrying a hidden state that summarizes what they have seen so far. Training uses backpropagation through time. A plain RNN has difficulty learning long-range dependencies because gradients are multiplied many times as they flow back through the steps and tend to shrink toward zero. This is the vanishing gradient problem, and the network effectively forgets early inputs. The LSTM, or long short-term memory network, addresses this with a separate cell state and three gates. The forget gate decides what to erase from the cell state, the input gate decides what new information to write, and the output gate decides what part of the cell state to expose as the hidden state. Because the cell state can pass information forward with little change, gradients survive over longer spans. GRUs are a lighter variant with two gates."),
    ("Transformers and Attention",
     "The transformer architecture replaces recurrence with self-attention. For each token, the model creates a query, a key and a value vector. The similarity between a token's query and every other token's key determines attention weights, and the output is the weighted sum of the value vectors. In this way each word can focus on the most relevant other words in the sentence, regardless of distance. Because attention has no built-in sense of order, positional encodings are added to the input embeddings. Multi-head attention runs several attention computations in parallel so the model can capture different relationships at once. Since all positions are processed simultaneously rather than step by step, transformers train much faster on modern hardware than recurrent networks. Large language models are built from stacks of transformer layers trained on huge text corpora to predict the next token."),
    ("Overfitting and Regularization",
     "Overfitting happens when a model learns the noise and quirks of the training set instead of the underlying pattern, so it performs well on training data but poorly on new data. A typical sign is a large gap between training accuracy and validation accuracy. Several remedies exist. Collecting more training data or using data augmentation gives the model more variety. Regularization adds a penalty to the loss: L2 regularization discourages large weights, while L1 regularization pushes many weights to exactly zero and so performs feature selection. In neural networks, dropout randomly switches off a fraction of neurons during each training step, which prevents units from relying too heavily on each other. Early stopping monitors the validation loss and halts training when it begins to rise. Simplifying the model, for example using fewer layers or a shallower tree, also helps. Cross-validation gives a more reliable estimate of generalization."),
    ("Gradient Descent",
     "Gradient descent is the optimization algorithm that trains most machine learning models. It computes the gradient of the loss with respect to the parameters and moves the parameters a small step in the opposite direction. The size of the step is set by the learning rate. If the learning rate is too small, training is slow; if it is too large, the updates overshoot the minimum and the loss may oscillate or even diverge. Batch gradient descent uses the whole dataset for every update, which is accurate but slow. Stochastic gradient descent, or SGD, updates the parameters using a single example, and mini-batch gradient descent uses a small group of examples, trading a little noise for much faster progress. Momentum accumulates past gradients to smooth the updates. Adam adapts the learning rate for each parameter using running averages of the gradients and their squares, and is a popular default."),
]

# (question, gold page). Several are deliberately paraphrased so they share few
# words with the notes; keyword search should struggle on those and meaning-based
# (dense) search should help. That contrast is what the project measures.
QUESTIONS = [
    ("What does the least squares method minimize?", 1),
    ("How do we measure how far predictions are from the true values in a regression model?", 1),
    ("Why can accuracy be misleading when one class is much rarer than the others?", 2),
    ("Which metric combines precision and recall into one number?", 2),
    ("What criterion decides where a decision tree splits the data?", 3),
    ("How does combining many trees reduce overfitting compared to a single tree?", 3),
    ("Why do image models share the same small filter across the whole picture?", 4),
    ("What does pooling do to feature maps?", 4),
    ("Why do plain recurrent networks forget information from many steps ago?", 5),
    ("What are the three gates in an LSTM cell?", 5),
    ("How does a model decide which words in a sentence to focus on?", 6),
    ("Why are transformers faster to train than recurrent networks?", 6),
    ("What can I do when training accuracy is high but validation accuracy is poor?", 7),
    ("What does dropout do during training?", 7),
    ("What happens if the step size during optimization is too large?", 8),
    ("How is stochastic gradient descent different from using the full dataset each step?", 8),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for index, (title, body) in enumerate(PAGES):
        story += [Paragraph(title, styles["Heading1"]), Spacer(1, 12), Paragraph(body, styles["BodyText"])]
        if index < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT / "ml_notes.pdf"), pagesize=A4).build(story)

    with open(OUT / "qa.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "source", "page"])
        for question, page in QUESTIONS:
            writer.writerow([question, "ml_notes.pdf", page])
    print(f"Wrote {OUT / 'ml_notes.pdf'} and {OUT / 'qa.csv'}")


if __name__ == "__main__":
    main()
