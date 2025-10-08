# Script complet
import streamlit as st
import requests

# --- Configuració ---
API_URL = "http://localhost:8000/query"  # L'endpoint del teu backend FastAPI
st.set_page_config(page_title="Xat RAG", page_icon="🤖")

# --- Títol de l'Aplicació ---
st.title("🤖 Xat amb els teus Documents")
st.caption("Aquesta interfície utilitza un model LLM local per respondre preguntes sobre la teva documentació.")

# --- Inicialització de l'Historial del Xat ---
# Streamlit re-executa l'script amb cada interacció.
# Hem de guardar l'historial a la 'session_state' per no perdre'l.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola! Fes-me una pregunta sobre els teus documents."}
    ]

# --- Mostrar Missatges Anteriors ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada de l'Usuari ---
if prompt := st.chat_input("Escriu la teva pregunta aquí..."):
    # Afegeix el missatge de l'usuari a l'historial i mostra'l
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Crida al Backend i Mostrar Resposta ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensant... 🧠")
        
        try:
            # Prepara les dades per a l'API
            payload = {"question": prompt}
            
            # Fes la petició POST al teu backend
            response = requests.post(API_URL, json=payload, timeout=300) # Timeout llarg per si el model triga
            response.raise_for_status()  # Llançarà un error si la resposta no és 2xx
            
            # Processa la resposta
            result = response.json()
            answer = result.get("answer", "No he rebut una resposta vàlida.")
            
            # Mostra la resposta
            message_placeholder.markdown(answer)
            
            # Afegeix la resposta de l'assistent a l'historial
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.RequestException as e:
            error_message = f"Error de connexió amb el backend: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})