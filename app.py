import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import hashlib
from pathlib import Path

import torch

torch.classes.__path__ = []

import streamlit as st
from openai import OpenAI

from config import CHAT_MODEL, FOUNDRY_BASE_URL, MAX_CONTEXT_CHARS_PER_CHUNK, MAX_OUTPUT_TOKENS, RAW_DIR
from document_loader import extract_document
from quality_gate import clean_answer, validate_answer
from rag_engine import clear_documents, delete_document, index_document, list_documents, retrieve

st.set_page_config(page_title="Field-Service Copilot", page_icon="🛠️", layout="wide")

SUPPORTED_TYPES = ["pdf", "txt", "md", "docx", "csv", "xlsx", "pptx", "html", "htm"]

GENERAL_PROMPT = """
Sen Türkçe konuşan, doğal ve güvenilir bir asistansın.
Türkçe sorulara yalnızca akıcı, doğru ve doğal Türkçe ile cevap ver. Dilbilgisi, noktalama ve anlam bütünlüğüne dikkat et.
Kullanıcı selam verirse sıcak ve kısa biçimde yanıtla. Tarih, yıl veya güncel olaylar hakkında bağlam yoksa kesin bilgi uydurma.
Sorunun kapsamına göre kısa ama yeterli açıklama yap; gerekirse maddeler kullan.
İç düşünce, analiz, plan, İngilizce sistem metni veya <think> etiketi yazma.
"""

DOCUMENT_PROMPT = """
Sen Türkçe konuşan, kaynak odaklı bir doküman asistanısın.
Yalnızca verilen kaynaklara dayan. Kaynakta olmayan ayrıntıları, sayıları veya tarihleri uydurma.
Yanıtı akıcı Türkçe ile yaz; aynı bilgiyi tekrar etme. Kaynaklar eksikse bunu açıkça belirt.
Yanıt sonunda `Kaynaklar:` başlığı altında dosya adı ve sayfa/bölüm bilgisini yaz.
İç düşünce, analiz, plan, İngilizce sistem metni veya <think> etiketi yazma.
"""


def is_general_chat(text: str) -> bool:
    normalized = text.lower().strip().rstrip("!?.")
    phrases = {
        "merhaba", "selam", "selamlar", "hey", "günaydın", "iyi günler",
        "iyi akşamlar", "nasılsın", "ne haber", "teşekkürler", "teşekkür ederim",
        "sağ ol", "sağol", "yardım", "ne yapabiliyorsun", "görüşürüz", "hoşça kal",
    }
    return normalized in phrases


def save_raw(uploaded_file) -> None:
    data = uploaded_file.getvalue()
    safe_name = Path(uploaded_file.name).name.replace("/", "_").replace("\\", "_")
    path = RAW_DIR / f"{hashlib.sha256(data).hexdigest()[:12]}_{safe_name}"
    if not path.exists():
        path.write_bytes(data)


def build_context(results: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[KAYNAK: {item['filename']} | {item['location']}]\n"
        f"{item['text'][:MAX_CONTEXT_CHARS_PER_CHUNK]}"
        for item in results
    )


def ask_model(system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(base_url=st.session_state.base_url, api_key="not-needed")
    response = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"/no_think\n{user_prompt}"},
        ],
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


def revise_answer(answer: str, source_context: str) -> str:
    prompt = f"""Aşağıdaki taslak yanıtı yalnızca tekrarları kaldırarak, cümleleri düzelterek ve daha akıcı Türkçe kullanarak yeniden yaz.
Yeni bilgi ekleme; sadece taslakta ve kaynaklarda geçen bilgilere dayan.

[TASLAK]\n{answer}\n\n[KAYNAKLAR]\n{source_context}"""
    return ask_model(DOCUMENT_PROMPT, prompt)


def index_upload(uploaded_file) -> dict:
    data = uploaded_file.getvalue()
    save_raw(uploaded_file)
    sections, metadata = extract_document(uploaded_file.name, data)
    if metadata.get("ocr_required", False):
        return {"error": "PDF'de okunabilir metin bulunamadı. Bu dosya taranmış olabilir; OCR gerekir."}
    if not sections:
        return {"error": "Dosyadan temiz ve indekslenebilir metin çıkarılamadı."}
    return {"data": data, "sections": sections, "metadata": metadata}


if "messages" not in st.session_state:
    st.session_state.messages = []
if "base_url" not in st.session_state:
    st.session_state.base_url = FOUNDRY_BASE_URL
if "model_name" not in st.session_state:
    st.session_state.model_name = CHAT_MODEL

st.title("🛠️ Field-Service Copilot")
st.caption("Yerel sohbet · Hibrit arama · Reranking · Kaynaklı yanıt")

with st.sidebar:
    st.header("Sistem")
    st.session_state.base_url = st.text_input("Foundry endpoint", value=st.session_state.base_url)
    st.session_state.model_name = st.text_input("Chat modeli", value=st.session_state.model_name)
    mode = st.radio(
        "Yanıt modu",
        ["Otomatik", "Genel sohbet", "Yalnızca doküman"],
        index=0,
        help="Otomatik mod selamlaşmaları genel sohbete gönderir; diğer sorularda indeksli dosyaları kullanır.",
    )

    st.divider()
    st.subheader("Bilgi tabanı")
    uploads = st.file_uploader("Doküman ekle", type=SUPPORTED_TYPES, accept_multiple_files=True)

    if uploads:
        for upload in uploads:
            key = f"upload_{upload.name}_{upload.size}"
            if key not in st.session_state:
                progress = st.progress(0, text=f"{upload.name} okunuyor...")
                try:
                    progress.progress(5, text="Dosya okunuyor ve gürültülü metin ayıklanıyor...")
                    prepared = index_upload(upload)
                    if "error" in prepared:
                        st.session_state[key] = prepared
                    else:
                        result = index_document(
                            upload.name,
                            prepared["data"],
                            prepared["sections"],
                            prepared["metadata"],
                            progress_callback=lambda value, text: progress.progress(value, text=text),
                        )
                        st.session_state[key] = {"result": result}
                except Exception as error:
                    st.session_state[key] = {"error": str(error)}
                finally:
                    progress.empty()

            status = st.session_state[key]
            if "error" in status:
                st.error(f"{upload.name}: {status['error']}")
            else:
                result = status["result"]
                suffix = " zaten indeksli." if result["already_indexed"] else " indekslendi."
                st.success(f"{upload.name}: {result['chunk_count']} temiz parça" + suffix)

    documents = list_documents()
    if documents:
        st.caption("İndeksli dosyalar")
        for document in documents:
            file_column, delete_column = st.columns([5, 1])
            file_column.write(f"• {document['filename']} ({document['chunk_count']} parça)")
            if delete_column.button("🗑️", key=f"delete_{document['document_id']}", help="Bu dokümanı indeksten kaldır"):
                delete_document(document["document_id"])
                st.session_state.messages = []
                for key in list(st.session_state.keys()):
                    if key.startswith("upload_"):
                        del st.session_state[key]
                st.rerun()

        if st.button("Tüm dokümanları indeksten kaldır", use_container_width=True):
            clear_documents()
            st.session_state.messages = []
            for key in list(st.session_state.keys()):
                if key.startswith("upload_"):
                    del st.session_state[key]
            st.rerun()
    else:
        st.caption("Henüz indekslenmiş doküman yok.")

    st.divider()
    if st.button("Sohbeti temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Sorunuzu yazın...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    documents = list_documents()
    use_documents = mode == "Yalnızca doküman" or (
        mode == "Otomatik" and bool(documents) and not is_general_chat(question)
    )
    results: list[dict] = []
    retrieval_error = ""

    if use_documents and documents:
        with st.spinner("Dokümanlarda aranıyor ve sonuçlar yeniden sıralanıyor..."):
            try:
                results = retrieve(question)
            except Exception as error:
                retrieval_error = str(error)

    with st.chat_message("assistant"):
        raw_answer = ""
        try:
            if mode == "Yalnızca doküman" and not documents:
                answer = "Doküman modu açık, ancak indekslenmiş bir dosya yok. Önce bir doküman yükleyin."
            elif use_documents and results:
                source_context = build_context(results)
                raw_answer = ask_model(DOCUMENT_PROMPT, f"[KAYNAKLAR]\n{source_context}\n\nKullanıcı sorusu: {question}")
                valid, answer = validate_answer(raw_answer, document_mode=True)
                if not valid and clean_answer(raw_answer):
                    revised = revise_answer(clean_answer(raw_answer), source_context)
                    _, answer = validate_answer(revised, document_mode=True)
            elif mode == "Yalnızca doküman":
                answer = "İndekslenmiş dokümanlarda bu soruyu destekleyecek yeterince yakın bir kaynak bulunamadı. Daha belirgin bir başlık, anahtar kelime veya bölüm adıyla tekrar deneyin."
            else:
                raw_answer = ask_model(GENERAL_PROMPT, question)
                valid, answer = validate_answer(raw_answer, document_mode=False)
                if not valid and clean_answer(raw_answer):
                    answer = clean_answer(raw_answer)

            st.markdown(answer)

            if use_documents:
                with st.expander("🔎 Arama tanısı"):
                    if raw_answer:
                        st.caption("Ham model yanıtı")
                        st.code(raw_answer)
                    if results:
                        for item in results:
                            st.markdown(f"**{item['filename']} — {item['location']}**")
                            st.caption(f"Semantic: {item['semantic_score']} | BM25: {item['bm25_score']} | Rerank: {item['rerank_score']}")
                            st.code(item["text"][:500])
                    elif retrieval_error:
                        st.error(f"Arama hatası: {retrieval_error}")
                    else:
                        st.info("Uygun kaynak bulunamadı.")
        except Exception as error:
            answer = "Yerel model şu anda yanıt üretemedi. Lütfen birkaç saniye sonra tekrar deneyin."
            st.error(answer)
            st.code(str(error))

        st.session_state.messages.append({"role": "assistant", "content": answer})