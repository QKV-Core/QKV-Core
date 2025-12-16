from .trainer import Trainer
from model_registry import ModelRegistry
import os
import time

class IncrementalTrainer(Trainer):
    
    def __init__(self, model, config, tokenizer, db_manager, device=None):
        super().__init__(model, config, tokenizer, db_manager, device)
        self.model_registry = ModelRegistry()
        self.research_corpus_paths = []
        self.research_citations = []
    
    def add_research_data(self, corpus_path: str, citation: str = ""):
        self.research_corpus_paths.append(corpus_path)
        self.research_citations.append(citation)
    
    def train_incremental(self, parent_model_id: str, run_name: str = "incremental_training"):
        
        if not self.research_corpus_paths:
            raise ValueError("No research data added for incremental training")
        
        combined_corpus = []
        for corpus_path in self.research_corpus_paths:
            if os.path.exists(corpus_path):
                # "if" keyword cannot be variable name, changed to "f"
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    combined_corpus.extend(f.read().split('\n'))
        
        from .dataset import TextDataset
        dataset = TextDataset(combined_corpus, self.tokenizer, 
                            max_length=self.config.get('max_seq_length', 512))
        
        print(f"🚀 Starting incremental training with {len(combined_corpus)} research samples")
        self.train(dataset, session_name=run_name)
        
        new_model_id = f"{parent_model_id}_inc_{int(time.time())}"
        model_entry = {
            'model_id': new_model_id,
            'model_path': f'model_weights/{new_model_id}.pt',
            'model_type': 'incremental-finetune',
            'parent_model': parent_model_id,
            'config': self.config,
            'train_date': time.strftime('%Y-%m-%d'),
            'source_data': self.research_corpus_paths,
            'citation': self.research_citations,
            'final_loss': self.best_loss
        }
        
        self.model_registry.add_model(model_entry)
        print(f"✅ Incremental training complete! New model registered: {new_model_id}")
        
        return new_model_id