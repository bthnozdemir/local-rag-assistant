# 🛠️ Offline Field-Service RAG Assistant

Microsoft Foundry Local ve RAG (Retrieval-Augmented Generation) mimarisi kullanılarak geliştirilmiş, tamamen çevrimdışı çalışan yerel doküman soru-cevap asistanı.

## 🚀 Özellikler
- **Tam Çevrimdışı Çalışma:** Bulut servislerine bağımlı olmadan yerel LLM inference.
- **Hibrit Arama (Hybrid Search):** BM25 ve Yoğun Vektör Arama (Dense Vector Search) kombinasyonu (RRF Fusion).
- **Cross-Encoder Reranking:** Arama sonuçlarını mmarco-mMiniLMv2 modeli ile yeniden sıralama.
- **Çoklu Doküman Desteği:** PDF, DOCX, XLSX, PPTX, CSV, HTML ve TXT formatlarından metin ayıklama.
- **Kalite Kapısı (Quality Gate):** Düşünme etiketleri temizleme (<think>), halüsinasyon ve tekrar engelleme.

## 🛠️ Kurulum

`ash
git clone [https://github.com/bthnozdemir/local-rag-assistant.git](https://github.com/bthnozdemir/local-rag-assistant.git)
cd local-rag-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
`

## 🚀 Çalıştırma

1. Foundry Local sunucusunu başlatın:
`ash
foundry server start
foundry model load qwen3-4b-cuda-gpu:2
`

2. Uygulamayı çalıştırın:
`ash
streamlit run app.py
`
