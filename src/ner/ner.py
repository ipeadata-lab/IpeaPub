from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from typing import List, Dict, Any

from src.config import MODELO_NER

class ModeloNER:
    def __init__(self, modelo: str = MODELO_NER):
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModelForTokenClassification.from_pretrained(modelo)
        self.pipeline = pipeline("ner", model=self.modelo, tokenizer=self.tokenizer, aggregation_strategy="first")
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo.to(self.dispositivo)
        self.id2label = self.modelo.config.id2label
        print(f"Modelo NER carregado: {modelo}")
    
    def extrair_entidades(self, texto: str) -> Dict[str, List]:
        """
        Extrai entidades nomeadas de um texto usando o modelo NER.
        
        Args:
            texto: Texto a ser analisado.
            
        Returns:
            Lista de dicionários com as entidades extraídas e suas respectivas classes.
        """
        if len(texto) > 5000:
            texto = texto[:5000]  # Limitar o tamanho do texto para evitar problemas de memória

        entidades_cruas = self.pipeline(texto)
        entidades = {}
        for entidade in entidades_cruas:
            if len(entidade["word"]) < 2 and not entidade["word"].isupper():
                continue
            if entidade.get("score", 0) < 0.8:
                continue
            tipo_entidade = entidade["entity_group"]
            if tipo_entidade not in entidades:
                entidades[tipo_entidade] = set()
            entidades[tipo_entidade].add(entidade["word"])

        for tipo_entidade in entidades:
            entidades[tipo_entidade] = list(entidades[tipo_entidade])
        return entidades
    
    def enriquecer_metadados(self, texto: str, metadados: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriquecer os metadados com as entidades extraídas do texto.
        
        Args:
            texto: Texto a ser analisado.
            metadados: Dicionário de metadados existentes.
            
        Returns:
            Dicionário de metadados enriquecido.
        """

        if len(texto.split()) > 20:
            try:
                entidades = self.extrair_entidades(texto)
                if entidades:
                    metadados.update({
                        "entidades": entidades
                    })
            except Exception as e:
                print(f"Erro ao extrair entidades: {e}")

        return metadados
