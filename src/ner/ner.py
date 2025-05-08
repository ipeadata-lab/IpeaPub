from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import torch.nn.functional as F  # Import for softmax

from src.config import MODELO_NER

class ModeloNER:
    def __init__(self, modelo: str = MODELO_NER):
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModelForTokenClassification.from_pretrained(modelo)
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo.to(self.dispositivo)
        self.id2label = self.modelo.config.id2label
        print(f"Modelo NER carregado: {modelo}")
    
    def extrair_entidades(self, texto: str) -> List[Dict[str, Any]]:
        """
        Extrai entidades nomeadas de um texto usando o modelo NER.
        
        Args:
            texto: Texto a ser analisado.
            
        Returns:
            Lista de dicionários com as entidades extraídas e suas respectivas classes.
        """
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True, max_length = 512).to(self.dispositivo)
        
        with torch.no_grad():
            outputs = self.modelo(**inputs)

        predicoes = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entidades = []
        entidade_atual = None
        
        for token, predicao, logit in zip(tokens, predicoes[0], outputs.logits[0]):
            confidence = F.softmax(logit, dim=0)[predicao].item()  # Calculate confidence score

            if token.startswith("##"):
                if entidade_atual:
                    entidade_atual["texto"] += token[2:]
            else:
                if entidade_atual:
                    if entidade_atual["confidencia"] >= 0.8:  # Filter by confidence threshold
                        entidades.append(entidade_atual)
                    entidade_atual = None

                label = self.id2label[predicao.item()]
                if label != "O":
                    entidade_tipo = label.split("-")[1] if "-" in label else label
                    entidade_atual = {
                        "texto": token,
                        "classe": entidade_tipo,
                        "confidencia": confidence  # Add confidence score
                    }

        if entidade_atual and entidade_atual["confidencia"] >= 0.8:  # Final check for confidence
            entidades.append(entidade_atual)

        if entidade_atual:
            entidades.append(entidade_atual)

        entidades_processadas = self.processar_entidades(entidades, texto)
        return entidades_processadas
    
    def processar_entidades(self, entidades: List[Dict[str, Any]], texto: str) -> List[Dict[str, Any]]:
        """
        Processa as entidades extraídas para incluir informações adicionais.
        
        Args:
            entidades: Lista de entidades extraídas.
            texto: Texto original.
            
        Returns:
            Lista de dicionários com as entidades processadas.
        """
        processadas = []

        for entidade in entidades:
            texto_entidade = entidade["texto"].replace("##", "")
            if texto_entidade in texto:
                start_idx = texto.find(texto_entidade)
                processadas.append({
                    "texto": texto_entidade,
                    "tipo": entidade["classe"],
                    "inicio": start_idx,
                    "fim": start_idx + len(texto_entidade)
                })
        
        return processadas
    
    def enriquecer_metadados(self, texto: str, metadados: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriquecer os metadados com as entidades extraídas do texto.
        
        Args:
            texto: Texto a ser analisado.
            metadados: Dicionário de metadados existentes.
            
        Returns:
            Dicionário de metadados enriquecido.
        """
        entidades = self.extrair_entidades(texto)

        grupos = {}
        for entidade in entidades:
            tipo = entidade["tipo"]
            if tipo not in grupos:
                grupos[tipo] = []
            if entidade["texto"] not in grupos[tipo]:
                grupos[tipo].append(entidade["texto"])

        metadados_enriquecidos = metadados.copy()
        metadados_enriquecidos["entidades"] = grupos

        return metadados_enriquecidos
