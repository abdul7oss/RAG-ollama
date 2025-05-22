import streamlit as st
import fitz  # PyMuPDF
import base64

def save_uploaded_file(uploaded_file):
    """Save uploaded PDF to a temporary file."""
    file_path = "temp.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path



uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    pdf_path = save_uploaded_file(uploaded_file)

    # User Inputs
    page_number = st.number_input("Enter Page Number", min_value=1, step=1, value=1)
    x0 = st.number_input("x0 (Left)", min_value=0, step=10, value=0)
    y0 = st.number_input("y0 (Bottom)", min_value=0, step=10, value=0)
    x1 = st.number_input("x1 (Right)", min_value=0, step=10, value=100)
    y1 = st.number_input("y1 (Top)", min_value=0, step=10, value=100)

    # Convert PDF to Base64 (for embedding in HTML)
    with open(pdf_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Scroll to the given rectangle
    y_scroll = y1  # PDF coordinates: (0,0) is bottom-left

    pdf_viewer_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.min.js"></script>
        <script>
            function loadPDF() {{
                var url = "data:application/pdf;base64,{pdf_base64}";
                var loadingTask = pdfjsLib.getDocument(url);
                loadingTask.promise.then(function(pdf) {{
                    pdf.getPage({page_number}).then(function(page) {{
                        var scale = 1.5;
                        var viewport = page.getViewport({{scale: scale}});

                        var canvas = document.getElementById("pdf-canvas");
                        var context = canvas.getContext("2d");
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;

                        var renderContext = {{
                            canvasContext: context,
                            viewport: viewport
                        }};
                        page.render(renderContext).promise.then(function() {{
                            setTimeout(function() {{
                                window.scrollTo(0, {y_scroll});  // Scroll to rectangle
                            }}, 500);
                        }});
                    }});
                }});
            }}

            window.onload = loadPDF;
        </script>
    </head>
    <body>
        <canvas id="pdf-canvas"></canvas>
    </body>
    </html>
    """

    st.components.v1.html(pdf_viewer_html, height=900)
