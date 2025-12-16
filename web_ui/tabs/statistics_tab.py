"""
Statistics tab for training metrics and analytics.
"""
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from ..state.app_state import state


def create_statistics_tab():
    """Create the statistics tab."""
    
    with gr.Tab("📊 Statistics"):
        gr.Markdown("# Training Statistics & Analytics")

        with gr.Tabs():
            with gr.Tab("📈 Training Metrics"):
                refresh_metrics_btn = gr.Button("🔄 Refresh Metrics")
                
                session_dropdown = gr.Dropdown(
                    label="Training Session select",
                    choices=[],
                    interactive=True
                )
                
                metrics_plot = gr.Plot(label="Loss Over Time")
                metrics_table = gr.Dataframe(label="Recent Metrics")
                
                def get_training_sessions():
                    return []
                
                def plot_metrics(session_id):
                    if not session_id:
                        return None, None
                    
                    metrics = state.db.get_training_metrics(session_id, limit=1000)
                    
                    if not metrics:
                        return None, None
                    
                    df = pd.DataFrame(metrics)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['loss'],
                        mode='lines',
                        name='Loss'
                    ))
                    
                    fig.update_layout(
                        title='Training Loss',
                        xaxis_title='Step',
                        yaxis_title='Loss',
                        template='plotly_white'
                    )
                    
                    return fig, df[['step', 'epoch', 'loss', 'learning_rate']].tail(20)
                
                session_dropdown.change(
                    fn=plot_metrics,
                    inputs=session_dropdown,
                    outputs=[metrics_plot, metrics_table]
                )
            
            with gr.Tab("🗂️ Models"):
                refresh_models_btn = gr.Button("🔄 Refresh Models")
                models_table = gr.Dataframe(label="Available Models")
                
                def get_models():
                    versions = state.db.list_model_versions()
                    if not versions:
                        return None
                    
                    df = pd.DataFrame(versions)
                    return df[['id', 'version_name', 'd_model', 'num_layers', 'num_heads', 'total_parameters', 'created_at']]
                
                refresh_models_btn.click(
                    fn=get_models,
                    outputs=models_table
                )
                
                models_table.value = get_models()
            
            with gr.Tab("💾 Storage Info"):
                storage_info = gr.Textbox(label="Storage Statistics", lines=15, interactive=False)
                
                def get_storage_info():
                    stats = state.db.get_statistics()
                    
                    info = f"""Storage Statistics:
• Total Models: {stats.get('total_models', 0)}
• Total Training Sessions: {stats.get('total_sessions', 0)}
• Storage Used: {stats.get('storage_used', 'N/A')}
• Database Size: {stats.get('db_size', 'N/A')}"""

                    if 'latest_model' in stats and stats['latest_model']:
                        latest = stats['latest_model']
                        info += f"\n• Latest Model: {latest.get('name', 'N/A')} ({latest.get('created_at', 'N/A')})"

                    return info

                refresh_storage_btn = gr.Button("🔄 Refresh")
                refresh_storage_btn.click(
                    fn=get_storage_info,
                    outputs=storage_info
                )
                
                storage_info.value = get_storage_info()
