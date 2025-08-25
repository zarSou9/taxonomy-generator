1. Go to https://arxiv.org/category_taxonomy and find the category you want to scrape
2. Verify the category code exists in `category_metadata/arxiv_categories.json`. If it doesn't it may be out of date, use `scripts/arxiv_categories_to_json.py` to update it
3. Set the `CATEGORY` in your `.env`
4. Set placeholder values for the category in `category_metadata/corpus_cuttofs.json`
5. Run `scripts/fetch_papers_under_category.py` to fetch the papers
6. Run `scripts/fill_paper_citation_counts.py` to fetch the citation counts
7. Run `scripts/analyze_arxiv_cat_citation_counts.py` to analyze the citation cuttoffs, and tune them in `category_metadata/corpus_cuttofs.json` as necessary. Example:

```json5
{
    "citation_cutoffs": {
        "0": 1,  // Papers published in 2025 must have at least one citation
        "1": 4,  // Papers published in 2024 must have at least 4 citations
        "3": 20,  // Papers published in 2022-2023 must have at least 4 citations
        "5": 60,  // ...
        "7": 100,  // ...
        "10": 140,  // ...
        "15": 200,  // ...
        "-1": 250  // Papers published in 1991-2009 require at least 250 citations
    },
    "year_start": 1991,
    "year_end": 2025
}
```

8. Run `scripts/filter_arxiv_papers.py` to filter the papers based on the citation cuttoffs
9. Run `scripts/ai_filter_arxiv_papers.py` to filter the papers based on an AI quality/relevance filter
