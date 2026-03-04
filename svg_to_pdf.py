from cairosvg import svg2pdf
import os

def svg_to_pdf(svg_path: str, pdf_path: str):
    """Convert SVG → PDF giữ nguyên width/height/viewBox"""
    svg2pdf(
        url=svg_path,      # Input SVG
        write_to=pdf_path, # Output PDF
        # 🔥 TỰ ĐỘNG giữ nguyên page size từ SVG attributes
        scale=1.0,         # Không scale
        dpi=72.0,          # Standard PDF DPI (không ảnh hưởng size)
    )
    print(f"✅ {svg_path} → {pdf_path} (kept original size)")

# Sử dụng
svg_to_pdf(r"tool_extract\static\outputs\b2d5d38b9cf34f77a8812b534b6e2d57_hybrid.svg", "output.pdf")
