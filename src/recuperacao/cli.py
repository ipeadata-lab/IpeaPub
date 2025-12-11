"""
Interface CLI para testar a pipeline de agentes.
Execute com: python -m src.recuperacao.cli
"""

import argparse
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG IPEA - Interface de Linha de Comando"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query para buscar (modo interativo se não fornecida)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Modelo OpenAI a utilizar (padrão: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Modo de execução: 'quick' (agente único) ou 'full' (pipeline completa)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Modo interativo (conversa contínua)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Habilita saída detalhada para depuração"
    )

    args = parser.parse_args()
    
    # Importar após parse para evitar erros se só pedir help
    from src.recuperacao.pipeline import RAGPipeline, create_coordinator_agent
    
    if args.interactive or not args.query:
        # Modo interativo
        print("\n" + "="*60)
        print("  Pipeline RAG IPEA - Modo Interativo")
        print("  Modelo:", args.model)
        print("  Digite 'sair' para encerrar")
        print("="*60 + "\n")
        
        if args.mode == "full":
            pipeline = RAGPipeline(model_id=args.model, verbose=args.verbose)
            
            while True:
                try:
                    query = input("\n🔍 Sua pergunta: ").strip()
                    
                    if query.lower() in ["sair", "exit", "quit", "q"]:
                        print("\nAté logo!")
                        break
                    
                    if not query:
                        continue
                    
                    result = pipeline.run(query)
                    
                    print("\n" + "-"*60)
                    print("📝 RESPOSTA:")
                    print("-"*60)
                    print(result["response"])
                    
                    if result.get("sources"):
                        print("\n📚 Fontes:")
                        for source in result["sources"]:
                            print(f"  - {source}")
                    
                    print(f"\n✓ Confiança: {result.get('confidence', 'N/A')}")
                    
                except KeyboardInterrupt:
                    print("\n\nInterrompido pelo usuário.")
                    break
                except Exception as e:
                    print(f"\n❌ Erro: {e}")
        
        else:  # quick mode
            agent = create_coordinator_agent(args.model)
            
            while True:
                try:
                    query = input("\n🔍 Sua pergunta: ").strip()
                    
                    if query.lower() in ["sair", "exit", "quit", "q"]:
                        print("\nAté logo!")
                        break
                    
                    if not query:
                        continue
                    
                    response = agent.run(query)
                    
                    print("\n" + "-"*60)
                    print("📝 RESPOSTA:")
                    print("-"*60)
                    print(response.content)
                    
                except KeyboardInterrupt:
                    print("\n\nInterrompido pelo usuário.")
                    break
                except Exception as e:
                    print(f"\n❌ Erro: {e}")
    
    else:
        # Modo single query
        if args.mode == "full":
            from src.recuperacao.pipeline import run_full_pipeline
            result = run_full_pipeline(args.query, model_id=args.model)
            
            print("\n" + "="*60)
            print("📝 RESPOSTA:")
            print("="*60)
            print(result["response"])
            
            if result.get("sources"):
                print("\n📚 Fontes:")
                for source in result["sources"]:
                    print(f"  - {source}")
            
            print(f"\n✓ Confiança: {result.get('confidence', 'N/A')}")
        
        else:
            from src.recuperacao.pipeline import quick_search
            response = quick_search(args.query, model_id=args.model)
            
            print("\n" + "="*60)
            print("📝 RESPOSTA:")
            print("="*60)
            print(response)


if __name__ == "__main__":
    main()
