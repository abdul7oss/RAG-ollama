import sys
from PyPDF2 import PdfReader

def extract_annotations(pdf_path):
    """
    Extract annotations from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary with page numbers as keys and list of annotations as values
    """
    try:
                pdf_reader = PdfReader(pdf_path)
                
                # Process each page
                for i, page in enumerate(pdf_reader.pages):
                    
                    
                    # Extract annotations if they exist
                    if '/Annots' in page:
                        page_annotations = []
                        try:
                            annots = page['/Annots']
                            for annot in annots:
                                annot_obj = annot.get_object()
                                print(annot_obj.get('content'))
                                if annot_obj:
                                    annotation_text = annot_obj
                                    # Get annotation coordinates if available
                                    rect = annot_obj.get('/Rect', [0, 0, 0, 0])
                                    
                                    # Add annotation to our collection with page info
                                    page_annotations.append({
                                        'text': annotation_text,
                                        'page': i,
                                        'position': current_position + len(page_text) // 2,  # Approximate position in text
                                        'coordinates': rect
                                    })
                        except Exception as e:
                            logger.error(f"Error extracting annotations from page {i}: {e}")
                        
                        if page_annotations:
                            # Store annotations indexed by their position in the combined text
                            for annot in page_annotations:
                                annotations[annot['position']] = annot
                    
                    # Add the page text to the combined text
                    # text += page_text
                    # current_position += len(page_text)
                    # print(annotations)
                    
    except Exception as e:
        print(f"Error extracting annotations: {e}")
        return {}

def print_annotations(annotations):
    """
    Print extracted annotations in a readable format.
    
    Args:
        annotations (dict): Dictionary with page numbers as keys and annotations as values
    """
    if not annotations:
        print("No annotations found in the PDF.")
        return
        
    print("\n=== ANNOTATIONS EXTRACTED FROM PDF ===\n")
    
    for page_num, page_annotations in annotations.items():
        print(f"Page {page_num}:")
        print("-" * 40)
        
        for i, annot in enumerate(page_annotations, 1):
            print(f"Annotation #{i}:")
            print(f"  Type: {annot['type']}")

            print(annot)
            
            if annot['content']:
                print(f"  Content: {annot['content']}")
            
            if annot['author']:
                print(f"  Author: {annot['author']}")
                
            print()
        
        print()

if __name__ == "__main__":
    annotations = extract_annotations('/Users/razim/Downloads/2501.12948v1.pdf')
    # print_annotations(annotations)