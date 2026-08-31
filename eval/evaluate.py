import os
import sys
import time
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src import document_loader, rag_engine

SAMPLE_FILE_PATH = os.path.join("data", "raw", "sample_field_manual.txt")

# Çeldiricili ve Çoklu Modelli Test Dokümanı
SAMPLE_CONTENT = """==================================================
XYZ-5000 & XYZ-6000 ENDÜSTRİYEL SANTRİFÜJ POMPA KILAVUZU
==================================================

1. XYZ-5000 Genel Özellikler ve Bakım
- Model: XYZ-5000-HD
- Maksimum Çalışma Basıncı: 16 Bar
- Çalışma Sıcaklık Aralığı: -10°C ile +110°C arası
- 2000 Saatlik Bakım: Mekanik salmastra değişimi ve ana yağ haznesi değişimi zorunludur.
- Önerilen Yağ Tipi: ISO VG 68 Sentetik Endüstriyel Dişli Yağı.

2. XYZ-6000 Ağır Hizmet Tipi Özellikler (Çeldirici Model)
- Model: XYZ-6000-MAX
- Maksimum Çalışma Basıncı: 25 Bar
- Çalışma Sıcaklık Aralığı: -20°C ile +150°C arası
- 2000 Saatlik Bakım: Sadece rulman bloğu ve hidrolik vana contası değiştirilir.
- Önerilen Yağ Tipi: ISO VG 46 Yüksek Sıcaklık Yağı.

3. Garanti İstisnaları ve Kapsam Dışı Durumlar
- Susuz (kuru) çalıştırma kaynaklı salmastra hasarları garanti kapsamı dışındadır.
- Yetkisiz servis tarafından yapılan müdahaleler garantiyi geçersiz kılar.
- Orijinal olmayan yedek parça (örneğin Parça Kodu: PRT-9921) kullanımı hasarlarda sorumluluk kabul edilmez.
"""

# Literatür Kriterlerine Uygun Test Kümesi
EVAL_DATASET = [
    {
        "category": "Anlamsal Arama (Semantic)",
        "query": "2000 saatlik bakımda hangi parçaların değişimi zorunludur?",
        "expected_keywords": ["salmastra", "yağ"],
        "should_have_answer": True
    },
    {
        "category": "Kapsamlı Kod / BM25 Match",
        "query": "PRT-9921 parça kodlu ürün kullanımı garantiye etki eder mi?",
        "expected_keywords": ["garanti", "sorumluluk"],
        "should_have_answer": True
    },
    {
        "category": "Çeldirici Modeller (Hard Negative)",
        "query": "XYZ-6000 modelinde önerilen yağ tipi hangisidir?",
        "expected_keywords": ["iso vg 46", "vg 46"],
        "should_have_answer": True
    },
    {
        "category": "Yazım Hatalı Sorgu (Typo Tolerance)",
        "query": "2000 saatlk bakmda ne değisir?",
        "expected_keywords": ["salmastra", "yağ"],
        "should_have_answer": True
    },
    {
        "category": "Kapsam Dışı (Out of Domain)",
        "query": "Uzay mekiği motoru hidrojen yakıt basıncı kaç bardır?",
        "expected_keywords": [],
        "should_have_answer": False
    },
    {
        "category": "Dokümanda Olmayan Detay",
        "query": "XYZ-5000 santrifüj pompanın yıllık sigorta bedeli ne kadardır?",
        "expected_keywords": [],
        "should_have_answer": False
    }
]

def setup_and_index():
    os.makedirs(os.path.dirname(SAMPLE_FILE_PATH), exist_ok=True)
    with open(SAMPLE_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(SAMPLE_CONTENT)

    if hasattr(rag_engine, "clear_documents"):
        rag_engine.clear_documents()

    with open(SAMPLE_FILE_PATH, "rb") as f:
        file_bytes = f.read()

    try:
        extracted = document_loader.extract_document(SAMPLE_FILE_PATH)
        sections = extracted[0] if isinstance(extracted, tuple) else extracted
        metadata = extracted[1] if isinstance(extracted, tuple) and len(extracted) > 1 else {}
    except Exception:
        sections = [{"title": "Kılavuz", "text": SAMPLE_CONTENT}]
        metadata = {}

    if not metadata:
        metadata = {"filename": "sample_field_manual.txt", "file_type": "txt"}

    rag_engine.index_document(
        filename="sample_field_manual.txt",
        data=file_bytes,
        sections=sections,
        metadata=metadata
    )

def run_benchmark():
    setup_and_index()
    
    print("="*65)
    print("📊 ADVANCED RAG BENCHMARK & EVALUATION REPORT")
    print("="*65 + "\n")

    total_queries = len(EVAL_DATASET)
    hit_count = 0
    correct_rejection_count = 0
    total_time = 0
    reciprocal_ranks = []
    unanswerable_queries = 0

    for idx, item in enumerate(EVAL_DATASET, 1):
        query = item["query"]
        should_answer = item["should_have_answer"]
        category = item["category"]
        
        start_time = time.time()
        results = rag_engine.retrieve(query)
        elapsed = time.time() - start_time
        total_time += elapsed

        top_score = 0.0
        rank_found = 0
        
        if results:
            first_res = results[0]
            if isinstance(first_res, dict):
                top_score = first_res.get("rerank_score", first_res.get("score", 0.0))

        if should_answer:
            # MRR (Mean Reciprocal Rank) Hesaplama
            for r_idx, res in enumerate(results, 1):
                chunk_text = res.get("text", res.get("content", "")).lower() if isinstance(res, dict) else str(res).lower()
                if any(kw.lower() in chunk_text for kw in item["expected_keywords"]):
                    rank_found = r_idx
                    break
            
            if rank_found > 0:
                hit_count += 1
                rr = 1.0 / rank_found
                reciprocal_ranks.append(rr)
                status = f"✅ HIT (Sıra #{rank_found})"
            else:
                reciprocal_ranks.append(0.0)
                status = "❌ MISS"
                
            print(f"[{idx}/{total_queries}] [{category}] {status} | Süre: {elapsed:.3f}s | Puan: {top_score:.3f}")
        else:
            unanswerable_queries += 1
            is_rejected = top_score < 0.25 or len(results) == 0
            if is_rejected:
                correct_rejection_count += 1
                status = "✅ REJECTED"
            else:
                status = "⚠️ UNCERTAIN PASS"
            print(f"[{idx}/{total_queries}] [{category}] {status} | Süre: {elapsed:.3f}s | Puan: {top_score:.3f}")

    answerable_count = total_queries - unanswerable_queries
    hit_rate = (hit_count / answerable_count * 100) if answerable_count > 0 else 0
    mrr_score = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0
    rejection_rate = (correct_rejection_count / unanswerable_queries * 100) if unanswerable_queries > 0 else 0
    avg_latency = total_time / total_queries if total_queries > 0 else 0

    print("\n" + "-"*65)
    print("📈 ÖLÇÜM SONUÇLARI METRİKLERİ (LİTERATÜR STANDARTLARI)")
    print("-"*65)
    print(f"• Getirme İsabet Oranı (Hit Rate) : %{hit_rate:.1f}")
    print(f"• Sıralama Kalitesi (MRR@K)       : {mrr_score:.3f} / 1.000")
    print(f"• Doğru Ret Başarısı (Rejection)  : %{rejection_rate:.1f}")
    print(f"• Ortalama Arama Süresi (Latency) : {avg_latency:.3f} saniye")
    print("="*65 + "\n")

if __name__ == "__main__":
    run_benchmark()