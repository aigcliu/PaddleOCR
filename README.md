<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="PaddleOCR Banner">
  </p>

  <h1>PaddleOCR</h1>

  <p><strong>Industry-leading, production-ready OCR and document AI engine</strong></p>
  <p>End-to-end solutions from text extraction to intelligent document understanding</p>

  <!-- Language Links -->
  <p>
    English | <a href="./readme/README_cn.md">简体中文</a> | <a href="./readme/README_tcn.md">繁體中文</a> | <a href="./readme/README_ja.md">日本語</a> | <a href="./readme/README_ko.md">한국어</a> | <a href="./readme/README_fr.md">Français</a> | <a href="./readme/README_ru.md">Русский</a> | <a href="./readme/README_es.md">Español</a> | <a href="./readme/README_ar.md">العربية</a>
  </p>

  <!-- Key Badges -->
  <p>
    <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/github/forks/PaddlePaddle/PaddleOCR.svg"></a>
    <a href="https://pypi.org/project/paddleocr/"><img src="https://img.shields.io/pypi/v/paddleocr"></a>
    <a href="https://pypi.org/project/paddleocr/"><img src="https://static.pepy.tech/badge/paddleocr/month"></a>
    <img src="https://img.shields.io/badge/python-3.8~3.12-aff.svg">
    <img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg">
    <a href="../LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green"></a>
  </p>

  <!-- Important Links -->
  <p>
    <a href="https://www.paddleocr.com">🌐 Official Website</a> |
    <a href="https://arxiv.org/pdf/2507.05595">📄 Technical Report 3.0</a> |
    <a href="https://arxiv.org/abs/2510.14528">📄 PaddleOCR-VL Paper</a> |
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/">📖 Documentation</a>
  </p>
</div>

---

## 🎯 What is PaddleOCR?

**PaddleOCR** converts documents and images into **structured, AI-friendly data** (like JSON and Markdown) with **industry-leading accuracy**—powering AI applications for everyone from indie developers and startups to large enterprises worldwide.

With over **60,000 stars** and deep integration into leading projects like **MinerU, RAGFlow, pathway and cherry-studio**, PaddleOCR has become the **premier solution** for developers building intelligent document applications in the **AI era**.

---

## ✨ Core Features

### 🚀 PaddleOCR 3.0 Highlights

| Feature | Description | Try It |
|---------|-------------|--------|
| **PaddleOCR-VL** | 0.9B multilingual VLM supporting 109 languages for document parsing | [🤗 HuggingFace](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo) • [🎨 AI Studio](https://aistudio.baidu.com/application/detail/98365) • [🔮 ModelScope](https://www.modelscope.cn/studios/PaddlePaddle/PaddleOCR-VL_Online_Demo) |
| **PP-OCRv5** | Universal scene text recognition supporting 5 languages with 13% accuracy boost | [🎨 Try Demo](https://aistudio.baidu.com/community/app/91660/webUI) |
| **PP-StructureV3** | Complex document parsing to Markdown/JSON, preserving original structure | [🎨 Try Demo](https://aistudio.baidu.com/community/app/518494/webUI) |
| **PP-ChatOCRv4** | Intelligent information extraction powered by ERNIE 4.5 | [🎨 Try Demo](https://aistudio.baidu.com/community/app/518493/webUI) |

### 🌟 Key Capabilities

- **🌍 Multilingual Support**: 100+ languages including Latin, Cyrillic, Arabic, Devanagari, and more
- **📊 Document Parsing**: Tables, formulas, charts, handwriting, historical documents
- **🎯 High Accuracy**: Industry-leading performance on public benchmarks
- **⚡ Fast Inference**: Optimized for CPU, GPU, XPU, and NPU
- **🔧 Production Ready**: Full toolkit for training, inference, and deployment
- **🐳 Multiple Deployment**: Python, C++, Serving, MCP Server, Mobile (Android)

---

## 🚀 Quick Start

### 1. Installation

**Requirements**: Python 3.8 - 3.12

```bash
# Install PaddleOCR
pip install paddleocr

# For GPU support (CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 📘 [Full Installation Guide](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/installation.html)

### 2. Basic Usage

#### **PP-OCRv5 - Text Recognition**

```python
from paddleocr import PPOCRv5

# Initialize pipeline
pipeline = PPOCRv5(use_angle_clf=True, lang="en", ocr_version="PP-OCRv5")

# Process image
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)

# View results
for res in output:
    res.print()
    res.save_to_img(save_path="output/")
    res.save_to_json(save_path="output/")
```

#### **PP-StructureV3 - Document Parsing**

```python
from paddleocr import PPStructureV3

# Initialize pipeline
pipeline = PPStructureV3()

# Parse document
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png"
)

# Save results
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

#### **PaddleOCR-VL - Vision-Language Model**

```python
from paddleocr import PaddleOCRVL

# Initialize pipeline
pipeline = PaddleOCRVL()

# Parse document
output = pipeline.predict(
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
)

# Save results
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

> 📘 More Examples: [PP-ChatOCRv4 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

---

## 📚 Documentation

### 📖 Tutorials

- [PP-OCRv5 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html) - Universal text recognition
- [PP-StructureV3 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html) - Document parsing
- [PP-ChatOCRv4 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html) - Information extraction
- [PaddleOCR-VL Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PaddleOCR-VL.html) - Vision-language model

### 🔧 Advanced Features

- [ONNX Model Export](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html)
- [High-Performance Inference](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html) (OpenVINO, TensorRT, ONNX Runtime)
- [Parallel Inference](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html) (Multi-GPU, Multi-process)
- [Serving Deployment](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html) (C++, C#, Java, etc.)
- [MCP Server](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html) (Integration with Claude Desktop)
- [Mobile Deployment](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html) (Android)

### 🖥️ Hardware Support

- [Huawei Ascend NPU](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
- [KUNLUNXIN XPU](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)

---

## 🔄 Demo Results

### PP-OCRv5
<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="PP-OCRv5 Demo">
  </p>
</div>

### PP-StructureV3
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="PP-StructureV3 Demo">
  </p>
</div>

### PaddleOCR-VL
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="PaddleOCR-VL Demo">
  </p>
</div>

---

## 📣 Recent Updates

<details>
<summary><strong>🔥 2025.10.16: PaddleOCR 3.3.0</strong></summary>

- **PaddleOCR-VL Release**:
  - 0.9B VLM supporting 109 languages
  - SOTA performance in document parsing
  - Available on [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

- **PP-OCRv5 Multilingual**:
  - Added support for 109 languages
  - Covers Latin, Cyrillic, Arabic, Devanagari, Telugu, Tamil
  - Only 2M parameters with 40%+ accuracy improvement

</details>

<details>
<summary><strong>2025.08.21: PaddleOCR 3.2.0</strong></summary>

- PP-OCRv5 English, Thai, Greek models
- Full support for PaddlePaddle 3.1.0/3.1.1
- C++ deployment for Linux and Windows
- CUDA 12 support with ONNX Runtime backend
- Fine-grained benchmarking tools

</details>

<details>
<summary><strong>2025.06.29: PaddleOCR 3.1.0</strong></summary>

- PP-OCRv5 multilingual models (37 languages)
- Upgraded PP-Chart2Table (9.36pp improvement)
- PP-DocTranslation pipeline with ERNIE 4.5
- MCP server support

</details>

> 📝 [View Full Changelog](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html)

---

## 🤝 Community & Support

<div align="center">

| WeChat Official Account | Tech Discussion Group |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |

</div>

### 💡 Get Help

- 📖 [Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/)
- 🐛 [GitHub Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)
- 💬 [Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions)

---

## 🌟 Awesome Projects Using PaddleOCR

<div align="center">

| Project | Description | Stars |
| ------- | ----------- | ----- |
| [RAGFlow](https://github.com/infiniflow/ragflow) | RAG engine based on deep document understanding | <img src="https://img.shields.io/github/stars/infiniflow/ragflow"> |
| [pathway](https://github.com/pathwaycom/pathway) | Python ETL framework for real-time analytics and LLM pipelines | <img src="https://img.shields.io/github/stars/pathwaycom/pathway"> |
| [MinerU](https://github.com/opendatalab/MinerU) | Multi-type Document to Markdown Conversion Tool | <img src="https://img.shields.io/github/stars/opendatalab/MinerU"> |
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) | Free, Open-source, Batch Offline OCR Software | <img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"> |
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) | Desktop client for multiple LLM providers | <img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"> |
| [OmniParser](https://github.com/microsoft/OmniParser) | Screen parsing tool for vision-based GUI agents | <img src="https://img.shields.io/github/stars/microsoft/OmniParser"> |

[📋 See More Projects →](./awesome_projects.md)

</div>

---

## 🙏 Contributors

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20" width="800"/>
</a>

**⭐ Star this repository to stay updated with new features and improvements! ⭐**

<img width="600" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star History">

</div>

---

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🎓 Citation

If you use PaddleOCR in your research, please cite:

```bibtex
@misc{cui2025paddleocr30technicalreport,
  title={PaddleOCR 3.0 Technical Report},
  author={Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
  year={2025},
  eprint={2507.05595},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.05595},
}

@misc{cui2025paddleocrvlboostingmultilingualdocument,
  title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
  author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
  year={2025},
  eprint={2510.14528},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.14528},
}
```

---

<div align="center">

**Built with ❤️ by the PaddlePaddle team**

[Official Website](https://www.paddleocr.com) • [Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/) • [GitHub](https://github.com/PaddlePaddle/PaddleOCR)

</div>
