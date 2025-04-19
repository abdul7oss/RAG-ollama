import streamlit as st
import fitz  # PyMuPDF
import base64
import urllib.parse




# Get query parameters
query_params = st.query_params
pdf_file = query_params.get("file", "temp.pdf")
page_number = int(query_params.get("page", 1))



# Convert PDF to Base64 (for embedding in HTML)
with open(pdf_file, "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

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

                    // Dynamically set the size
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;

                    var renderContext = {{
                        canvasContext: context,
                        viewport: viewport
                    }};
                    page.render(renderContext);

                    // Adjust the container size
                    document.getElementById("canvas-container").style.height = viewport.height + "px";
                    document.getElementById("canvas-container").style.width = "100%";
                }});
            }});
        }}

        window.onload = loadPDF;
    </script>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        #canvas-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100vh; /* Ensure it takes full height */
            overflow: auto;
        }}
        #pdf-canvas {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div id="canvas-container">
        <canvas id="pdf-canvas"></canvas>
    </div>
</body>
</html>
"""

# Increase the height in Streamlit
st.components.v1.html(pdf_viewer_html, height=1400)

