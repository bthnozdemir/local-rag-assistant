# ??? Offline Field-Service RAG Assistant

Microsoft Foundry Local ve RAG (Retrieval-Augmented Generation) mimarisi kullanlarak gelitirilmi, tamamen evrimd alan yerel dokman soru-cevap asistan.

## ?? zellikler
- **Tam evrimd alma:** Bulut servislerine baml olmadan yerel LLM inference.
- **Hibrit Arama (Hybrid Search):** BM25 ve Youn Vektr Arama (Dense Vector Search) kombinasyonu (RRF Fusion).
- **Cross-Encoder Reranking:** Arama sonularn mmarco-mMiniLMv2 modeli ile yeniden sralama.
- **oklu Dokman Destei:** PDF, DOCX, XLSX, PPTX, CSV, HTML ve TXT formatlarndan metin ayklama.
- **Kalite Kaps (Quality Gate):** Dnme etiketleri temizleme (<think>), halsinasyon ve tekrar engelleme.

## ??? Kurulum

`ash
git clone [https://github.com/bthnozdemir/local-rag-assistant.git](https://github.com/bthnozdemir/local-rag-assistant.git)
cd local-rag-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
`

## ?? altrma

1. Foundry Local sunucusunu balatn:
`ash
foundry server start
foundry model load qwen3-4b-cuda-gpu:2
`

2. Uygulamay altrn:
`ash
streamlit run app.py
`
