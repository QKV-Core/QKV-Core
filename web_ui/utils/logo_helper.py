"""
Logo helper utilities for QKV Core Web UI.
"""
from pathlib import Path


def get_logo_path():
    """
    Get the path to the QKV Core logo PNG file.
    
    Returns:
        Path: Path to logo file if exists, None otherwise
    """
    project_root = Path(__file__).parent.parent.parent
    # Try different possible logo file names
    possible_names = ["logo.png", "qkv_core_logo.png", "QKV_CORE_LOGO.png"]
    
    for name in possible_names:
        logo_path = project_root / "logo_designs" / name
        if logo_path.exists():
            return logo_path
    return None


def get_logo_html(width=None, show_title=True, show_subtitle=True):
    """
    Generate HTML for displaying the QKV Core logo.
    
    Args:
        width (int or None): Logo width in pixels. If None, auto-adjusts based on context (default: None)
        show_title (bool): Show "QKV Core" title (default: True)
        show_subtitle (bool): Show subtitle (default: True)
    
    Returns:
        str: HTML string for logo display
    """
    logo_path = get_logo_path()
    
    if logo_path:
        # Auto-adjust width if not specified
        if width is None:
            width = 300  # Default for header
        
        title_html = ""
        subtitle_html = ""
        
        if show_title:
            title_html = '<h1 style="margin: 10px 0; background: linear-gradient(135deg, #0066FF, #00D4FF, #FF6B35); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">QKV Core</h1>'
        
        if show_subtitle:
            subtitle_html = '<p style="color: #666; font-size: 14px;">Query-Key-Value Core - The Core of Transformer Intelligence</p>'
        
        # Use absolute path for Gradio file serving
        logo_str = str(logo_path).replace('\\', '/')
        
        return f"""
            <div style="text-align: center; padding: 20px 0;">
                <img src="/file={logo_str}" alt="QKV Core Logo" style="max-width: {width}px; width: 100%; height: auto; margin-bottom: 10px; object-fit: contain;">
                {title_html}
                {subtitle_html}
            </div>
        """
    else:
        # Fallback to text if logo not found
        title = "QKV Core" if show_title else ""
        subtitle = "Query-Key-Value Core - The Core of Transformer Intelligence" if show_subtitle else ""
        return f"# {title}\n**{subtitle}**" if title or subtitle else ""


def get_logo_image_component(width=300):
    """
    Get Gradio Image component for logo (if logo exists).
    
    Args:
        width (int): Logo width in pixels
    
    Returns:
        gradio.Image or None: Image component if logo exists
    """
    import gradio as gr
    
    logo_path = get_logo_path()
    if logo_path:
        return gr.Image(
            value=str(logo_path),
            label="",
            show_label=False,
            container=False,
            width=width
        )
    return None

