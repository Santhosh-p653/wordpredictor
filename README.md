# 📚 Pure PyTorch Multi-Dataset Word Predictor

An independent, lightweight Deep Learning project featuring a **2-Layer LSTM network** built entirely in **PyTorch** (completely free of TensorFlow dependencies). The model is uniquely trained on a blended stream of contrasting text corpora—**classic Shakespearean theatrical drama** and **modern Elasticsearch technical documentation**—allowing it to predict the next sequential word across distinct writing styles.

---

## 🚀 Live Demo Showcase

Below is a demonstration of the interactive **Gradio** interface processing inputs, handling out-of-vocabulary terms gracefully, and executing next-word tensor inferences in real-time.
<p align="center">
  <video src="https://github.com/user-attachments/assets/371c0e07-1d70-47a5-9523-6fd0b27eb6e4" title="LSTM Word Predictor Demo Walkthrough" width="850" autoplay loop muted></video>
</p>

> 💡 **Note:** If you are editing this README on GitHub, drag and drop your renamed `demo.mp4` video directly into the markdown editor to generate your unique cloud asset stream link, then paste it into the `src=""` attribute above.

---

## 🧠 Model Architecture & Features

Standard feed-forward neural networks fail to process contextual language because they lack sequential memory. This architecture overcomes that by leveraging recurrent state-tracking:

* **Pure Python Tokenization:** Implements a custom text preprocessor to build vocabulary frequency indexes and handle left-aligned pre-padding sequences natively without framework overhead.
* **Embedding Layer:** Transforms text tokens into dense continuous vectors ($128$-dimensional space) capturing semantic relationships.
* **Stacked LSTM Layers:** Utilizes a deeper $2$-layer LSTM network with $256$ hidden dimensions to track complex long-range textual context across varied domains.
* **Regularization (Dropout):** Integrates a `dropout=0.3` rate between sequential layers to prevent overfitting on specific dataset biases.
* **Gradio Web Interface:** Deploys a clean GUI complete with automated runtime example inputs for streamlined evaluation.

---

## 🛠️ Project Structure

```text
word-predictor/
├── assets/
│   └── demo.mp4               # Professional demo video recording
├── app.py                     # Main PyTorch & Gradio application script
├── requirements.txt           # Environment dependencies list
└── README.md                  # Project documentation
