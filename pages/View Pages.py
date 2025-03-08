import streamlit as st

def main():
    st.title("PDF Viewer")
    
    # Check if a specific page was requested via session state (from citations)
    if "nav_to_page" in st.session_state:
        selected_page = st.session_state.pop("nav_to_page")
    else:
        selected_page = 0
    
    if "pdf_pages" in st.session_state:
        # Create a dropdown for page selection
        total_pages = len(st.session_state["pdf_pages"])
        selected_page = min(selected_page, total_pages - 1)  # Ensure page number is valid
        
        selected_page = st.selectbox(
            "Go to page", 
            range(total_pages),
            index=selected_page,
            format_func=lambda x: f"Page {x+1} of {total_pages}"
        )
        
        # Add zoom control
        zoom = st.slider(
            "Zoom Level", 100, 1000, 500, 50, key="zoom_pdf")
        
        # Display the selected page
        st.image(st.session_state["pdf_pages"][selected_page], width=zoom)
        
        # Add navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous Page") and selected_page > 0:
                st.session_state["selected_page"] = selected_page - 1
                st.rerun()
        
        with col2:
            if st.button("Next Page") and selected_page < total_pages - 1:
                st.session_state["selected_page"] = selected_page + 1
                st.rerun()
    else:
        st.warning("No PDF has been uploaded. Please go back to the main page and upload a PDF.")

if __name__ == "__main__":
    main()