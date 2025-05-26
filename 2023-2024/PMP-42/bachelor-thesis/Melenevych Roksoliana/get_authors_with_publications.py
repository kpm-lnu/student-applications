import requests
from collections import defaultdict


def get_authors_with_publications(min_papers=10, max_authors=1500):
    search_url = "http://export.arxiv.org/api/query"
    authors_count = defaultdict(int)
    start = 0
    max_results = 5000
    total_results = 0
    authors_with_min_papers = []

    while len(authors_with_min_papers) < max_authors:
        params = {
            "search_query": "all",
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        response = requests.get(search_url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        xml_data = response.text
        entries = xml_data.split('<entry>')
        total_results += len(entries) - 1

        for entry in entries[1:]:
            author_tags = entry.split('<author>')
            for author_tag in author_tags[1:]:
                name_tag = author_tag.split('<name>')[1]
                author_name = name_tag.split('</name>')[0]
                authors_count[author_name] += 1

        for author, count in authors_count.items():
            if count > min_papers and author not in authors_with_min_papers:
                authors_with_min_papers.append(author)
            if len(authors_with_min_papers) >= max_authors:
                break

        start += max_results

        if total_results < start:
            break

    return authors_with_min_papers