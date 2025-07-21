# 🧬 Virtual Cell AI

A hybrid AI + systems biology project for modeling gene expression dynamics, incorporating omics features and graph-based relational structure across genes.

Inspired by the vision of building a **"Virtual Cell"**, this project combines modern machine learning architectures (GNNs, delta prediction, skip connections) with principles from classical systems biology.

---

## 📂 Project Structure

```

virtual-cell/
├── model/
│   ├── encoders.py        # Input feature encoders (e.g., MLPs)
│   ├── gnn\_layers.py      # Custom GNN layers (e.g., GeneGNN)
│   ├── decoders.py        # Output decoding (with delta & residual prediction)
│   └── full\_model.py      # Encoder → GNN → Decoder wrapper
├── utils/
│   └── graph\_utils.py     # Graph batching, adjacency handling, etc.
├── dataset.py             # Custom PyTorch dataset/dataloader for gene + graph data
├── train.py               # Training loop with logging and evaluation
├── requirements.txt       # Python dependencies
└── README.md              # This file

````

---

## 🚀 Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
````

2. **Prepare your dataset**

* Gene feature matrix (`[batch_size, num_genes, num_features]`)
* Optional: Sample-specific gene graphs (`[batch_size, num_genes, num_genes]` adjacency matrices)
* Target outputs: expression levels, states, or other properties.

3. **Run training**

```bash
python train.py
```

---

## 🧠 Key Features

* ✅ Modular encoder, GNN, and decoder blocks
* ✅ Delta prediction and residual skip connections
* ✅ Batch-aware graph support (custom adjacency per sample)
* ✅ Dataset support for multi-modal data (genes + graph)
* ✅ Clean, maintainable PyTorch code

---

## 📚 Scientific Inspiration

This project draws conceptual and theoretical motivation from foundational work in:

* **Systems Biology & Whole-Cell Modeling**

  * Karr et al., *Cell*, 2012: [DOI](https://doi.org/10.1016/j.cell.2012.05.044)
  * Thornburg et al., *Cell*, 2022: [DOI](https://doi.org/10.1016/j.cell.2022.08.001)
* **Differentiation Waves & Cell State Models**

  * Gordon et al., *BioSystems*, 2012; *J Tissue Eng Regen Med*, 2015

We acknowledge the pioneering insights of Dr. Richard “Dick” Gordon in computational biology and morphogenesis.

---

## 🧑‍💻 Project Author

**Syed Hussain Ather**
AI Team Lead @ Alter-Learning
AI Engineer @ AAK Telescience
Research Collaborator @ OREL (Orthogonal Research & Education Lab)
📫 \[Optional contact, e.g., LinkedIn or email]

---

## 🪪 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📌 Notes

This project is actively evolving. Contributions, ideas, and scientific collaborations are welcome — especially those bridging **AI and mechanistic modeling** in biology.

