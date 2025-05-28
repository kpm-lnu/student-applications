import fitz

def convert_pdf_to_text(pdf_path):
	doc = fitz.open(pdf_path)
	text = ""
	for page_num in range(len(doc)):
		page = doc.load_page(page_num)
		text += page.get_text()
	return text