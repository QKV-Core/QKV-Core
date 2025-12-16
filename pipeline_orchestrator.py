from research_ingestion import ResearchDataIngestor
from research_ingestion.auto_retrainer import ResearchAutoUpdater
from model_registry import ModelRegistry
from conversation_engine.hybrid_engine import HybridConversationEngine
import time
from pathlib import Path

def ingest_research(arxiv_id=None, pdf_path=None, csv_path=None, max_results=1):
    
    ingestor = ResearchDataIngestor()
    entries = []
    
    if arxiv_id:
        try:
            arxiv_entry = ingestor.ingest_arxiv(arxiv_id, max_results=max_results)
            if isinstance(arxiv_entry, list):
                entries.extend(arxiv_entry)
                print(f"[INFO] ✅ {len(arxiv_entry)} arXiv articles ingested")
            else:
                entries.append(arxiv_entry)
                print(f"[INFO] ✅ arXiv article ingested: {arxiv_entry['id']}")
        except Exception as e:
            print(f"[INFO] ❌ arXiv ingestion error: {e}")
    
    if pdf_path:
        try:
            pdf_entry = ingestor.ingest_pdf(pdf_path)
            entries.append(pdf_entry)
            print(f"[INFO] ✅ PDF ingested: {pdf_entry['id']}")
        except Exception as e:
            print(f"[INFO] ❌ PDF ingestion error: {e}")
    
    if csv_path:
        try:
            csv_entry = ingestor.ingest_csv(csv_path)
            entries.append(csv_entry)
            print(f"[INFO] ✅ CSV ingested: {csv_entry['id']}")
        except Exception as e:
            print(f"[INFO] ❌ CSV ingestion error: {e}")
    
    return entries

def run_training_step(parent_id, corpus_path, citation):
    
    print(f"[TRAINING] 🚀 Starting incremental training...")
    print(f"  Parent: {parent_id or 'from-scratch'}")
    print(f"  Corpus: {corpus_path}")
    print(f"  Citation: {citation}")
    
    updater = ResearchAutoUpdater()
    # Assuming _default_training is an internal method exposed for the orchestrator
    new_model_id = updater._default_training(parent_id, corpus_path, citation)
    
    if new_model_id:
        print(f"[TRAINING] ✅ Training completed: {new_model_id}")
    else:
        print(f"[TRAINING] ❌ Training failed")
    
    return new_model_id

def run_auto_train(polling_interval=300, parent_model_id=None):
    
    updater = ResearchAutoUpdater(polling_interval=polling_interval)
    
    trained_models = updater.run_once(
        train_callback=run_training_step,
        parent_model_id=parent_model_id
    )
    
    return trained_models

def hybrid_chat(model_ids, prompt, mode='ensemble', **kwargs):
    
    reg = ModelRegistry()
    engine = HybridConversationEngine(reg, default_mode=mode)
    
    print(f"[CHAT] 💬 Prompt: {prompt}")
    print(f"[CHAT] 🤖 Models: {model_ids}")
    print(f"[CHAT] 🎯 Mode: {mode}")
    
    response = engine.chat(
        prompt=prompt,
        model_ids=model_ids,
        mode=mode,
        **kwargs
    )
    
    if 'combined_response' in response:
        print(f"[CHAT] ✅ Combined Response:\n{response['combined_response']}")
    elif 'voted_response' in response:
        print(f"[CHAT] ✅ Voted Response:\n{response['voted_response']}")
    elif 'fallback_response' in response:
        print(f"[CHAT] ✅ Fallback Response:\n{response['fallback_response']}")
    else:
        print(f"[CHAT] ✅ Responses:\n{response.get('all_responses', response)}")
    
    return response

def list_available_models():
    
    reg = ModelRegistry()
    models = reg.list_models()
    
    print(f"\n📊 Available Models ({len(models)}):")
    print("=" * 60)
    for model in models:
        print(f"  • {model['model_id']}")
        print(f"    Type: {model.get('model_type', 'N/A')}")
        print(f"    Parent: {model.get('parent_model', 'None')}")
        print(f"    Date: {model.get('train_date', 'N/A')}")
        print()
    
    return models

def main_demo():
    
    print("=" * 60)
    print("🚀 LLM-Core Research Pipeline Demo")
    print("=" * 60)
    
    print("\n📋 Step 1: Checking available models...")
    models = list_available_models()
    
    if models:
        print("\n💬 Step 4: Testing hybrid chat...")
        model_ids = [models[0]['model_id']] if models else []
        if len(models) > 1:
            model_ids.append(models[1]['model_id'])
        
        prompt = "What are the latest trends in artificial intelligence?"
        hybrid_chat(model_ids, prompt, mode='ensemble')
    else:
        print("\n⚠️ No models available for chat. Train a model first!")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline demo completed!")
    print("=" * 60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'ingest':
            arxiv_id = sys.argv[2] if len(sys.argv) > 2 else None
            ingest_research(arxiv_id=arxiv_id)
        
        elif command == 'train':
            run_auto_train()
        
        elif command == 'chat':
            prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello!"
            model_ids = sys.argv[3:] if len(sys.argv) > 3 else []
            if not model_ids:
                models = list_available_models()
                model_ids = [m['model_id'] for m in models[:2]]
            hybrid_chat(model_ids, prompt)
        
        elif command == 'list':
            list_available_models()
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python pipeline_orchestrator.py [ingest|train|chat|list]")
    else:
        main_demo()