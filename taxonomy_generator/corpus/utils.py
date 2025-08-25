from taxonomy_generator.models.corpus import Paper


def get_pretty_paper(
    paper: Paper,
    keys: list[str] | None = ["title", "published", "summary"],
) -> str:
    result: list[str] = []

    if keys is None:
        keys = ["title", "id", "published", "summary", "source", "citation_count"]

    for key in keys:
        if key == "title":
            result.append(f"Title: {paper.title}")
        elif key == "id":
            result.append(f"ID: {paper.id}")
        elif key == "published":
            result.append(f"Published: {paper.published}")
        elif key == "summary":
            if paper.summary.type == "abstract":
                result.append(f"Abstract: {paper.summary.text}")
            elif paper.summary.type == "ai_summary":
                result.append(f"Summary: {paper.summary.text}")
        elif key == "source":
            result.append(f"Source: {paper.source}")
        elif key == "citation_count" and paper.citation_count is not None:
            result.append(f"Citation Count: {paper.citation_count}")

    return "\n".join(result)
