# Script complet
import asyncio
from llama_cpp import Llama

class LLMHandler:
    """
    Handler per carregar i interactuar amb un model GGUF localment
    utilitzant llama-cpp-python.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, verbose: bool = True):
        """
        Carrega el model GGUF des del path especificat en inicialitzar.
        """
        print(f"🚀 Carregant model des de: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,      # Mida de la finestra de context
            n_threads=None, # Automàticament detecta els fils de CPU
            verbose=verbose
            # Per a GPU, afegeix: n_gpu_layers=-1 (per carregar tot el model)
        )
        print("✅ Model carregat correctament.")

    def _chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """
        Executa una compleció de xat amb el model local. Aquesta és una operació bloquejant.
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=1024 # Pots ajustar aquest límit
        )
        return response['choices'][0]['message']['content']

    async def achat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Versió asíncrona de '_chat' per a FastAPI."""
        # Executa la funció bloquejant en un fil separat
        return await asyncio.to_thread(self._chat, messages, temperature)

    async def translate(self, text: str, target_language: str) -> str:
        """Tradueix text utilitzant el model local."""
        prompt = f"Translate the following text to {target_language}. ONLY output the translation and nothing else. Text: \"{text}\""
        messages = [{"role": "user", "content": prompt}]
        
        translated = await self.achat(messages, temperature=0.1)
        return translated.strip().strip('"')