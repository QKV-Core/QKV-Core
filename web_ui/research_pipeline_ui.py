import gradio as gr
import os

# Placeholder classes for missing modules
class ResearchDataIngestor:
    def ingest_arxiv(self, arxiv_id):
        return {"id": arxiv_id, "type": "arxiv", "status": "placeholder"}

    def ingest_pdf(self, pdf_path):
        return {"id": pdf_path, "type": "pdf", "status": "placeholder"}

    def ingest_csv(self, csv_path):
        return {"id": csv_path, "type": "csv", "status": "placeholder"}

class ResearchAutoUpdater:
    def run_once(self, train_callback=None, parent_model_id=None):
        return ["placeholder_model_1", "placeholder_model_2"]

class ModelRegistry:
    def list_models(self):
        return [{"model_id": "placeholder_model_1"}, {"model_id": "placeholder_model_2"}]

class ResearchPipelineUI:
    def __init__(self):
        self.ingestor = ResearchDataIngestor()
        self.registry = ModelRegistry()
        
    def create_research_tab(self):
        
        # Fixed "grado" to "gradio" and proper tab name
        with gr.Tab("🔬 Research Pipeline") as tab:
            gr.Markdown("## Research Data Pipeline")
            
            with gr.Row():
                with gr.Column():
                    # Fixed labels to proper English
                    arxiv_input = gr.Textbox(label="arXiv ID", placeholder="e.g., 2401.12345")
                    pdf_input = gr.File(label="Upload PDF File", file_types=[".pdf"])
                    csv_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
                    # Fixed button text
                    ingest_btn = gr.Button("📥 Add Data to System", variant="primary")
                
                with gr.Column():
                    # Fixed "Operaton Statusu" -> "Operation Status"
                    ingest_status = gr.Textbox(label="Operation Status", interactive=False)
                    
                    # Fixed dropdown and button labels
                    parent_model_select = gr.Dropdown(
                        choices=[],
                        label="Parent Model (Optional)",
                        value=None,
                        info="Select parent model for training (uses latest if empty)"
                    )
                    refresh_parent_btn = gr.Button("🔄 Refresh Parent Models", size="sm")
                    
                    auto_train_btn = gr.Button("🚀 Start Automatic Training", variant="secondary")
                    train_status = gr.Textbox(label="Training Status", interactive=False, lines=5)
            
            def ingest_data(arxiv_id, pdf_file, csv_file):
                try:
                    entries = []
                    if arxiv_id:
                        # Fixed "ngest" -> "ingest"
                        entry = self.ingestor.ingest_arxiv(arxiv_id)
                        entries.append(f"arXiv: {entry.get('id', arxiv_id)}")
                    if pdf_file:
                        entry = self.ingestor.ingest_pdf(pdf_file.name)
                        entries.append(f"PDF: {entry.get('id', 'File')}")
                    if csv_file:
                        entry = self.ingestor.ingest_csv(csv_file.name)
                        entries.append(f"CSV: {entry.get('id', 'File')}")
                    
                    if entries:
                        # Fixed ".jon" -> ".join" and f-string syntax
                        return f"✅ Success: {', '.join(entries)} added to system!"
                    else:
                        return "⚠️ Please enter at least one data source (arXiv, PDF, or CSV)."
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            def start_auto_training(parent_model_id=None):
                try:
                    updater = ResearchAutoUpdater()
                    
                    trained_models = updater.run_once(
                        train_callback=None,
                        parent_model_id=parent_model_id
                    )
                    
                    if trained_models:
                        # Fixed "\in" -> "\n"
                        return f"✅ {len(trained_models)} new models trained and added to registry:\n" + "\n".join([f"  • {md}" for md in trained_models])
                    else:
                        return "⚠️ No new research data found or training failed."
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"❌ Training error: {str(e)}"
            
            def refresh_parent_models():
                # Fixed "lst_models" -> "list_models"
                models = self.registry.list_models()
                model_ids = [m['model_id'] for m in models]
                # Fixed "avalable" -> "available"
                return gr.update(choices=model_ids if model_ids else ["No models available"])
            
            # Event Listener Connections
            # Fixed "clck" -> "click"
            ingest_btn.click(
                fn=ingest_data,
                inputs=[arxiv_input, pdf_input, csv_input],
                outputs=ingest_status
            )
            
            auto_train_btn.click(
                fn=start_auto_training,
                inputs=[parent_model_select],
                outputs=train_status
            )
            
            refresh_parent_btn.click(
                fn=refresh_parent_models,
                inputs=[],
                outputs=[parent_model_select]
            )
            
        return tab

def add_research_tab_to_app():
    pipeline_ui = ResearchPipelineUI()
    return pipeline_ui.create_research_tab()