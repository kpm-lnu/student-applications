import os
import arxiv
import re

def download_author_papers(author_name, num_max_results, download_dir='arxiv_papers'):
	if not os.path.exists(download_dir):
		os.makedirs(download_dir)
	
	client = arxiv.Client()
	
	search = arxiv.Search(
		query=f"au:{author_name}",
		max_results=num_max_results,
		sort_by=arxiv.SortCriterion.SubmittedDate
	)
	
	pdf_paths = []
	pattern = r'[^a-zA-Z0-9]'

	for result in client.results(search):
		pdf_url = result.pdf_url
		pdf_name = f"{result.entry_id.split('/')[-1]}.{result.title}"
		pdf_name = re.sub(pattern, '_', pdf_name) + '.pdf'
		pdf_path = os.path.join(download_dir, pdf_name)

		result.download_pdf(dirpath=download_dir, filename=pdf_name)
		
		pdf_paths.append(pdf_path)
	
	return pdf_paths