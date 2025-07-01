import json
import re
from typing import Any

from bs4 import BeautifulSoup, Tag


def clean_description(text: str) -> str:
    """Clean description text by removing extra whitespace and newlines."""
    if not text:
        return ""

    # Replace multiple whitespace characters (including newlines) with single spaces
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned


def parse_arxiv_categories_html(html_file_path: str) -> dict[str, Any]:
    """Parse the arXiv categories HTML file and convert to JSON structure.

    Args:
        html_file_path: Path to the HTML file containing arXiv categories

    Returns:
        Dictionary with structured category data
    """
    with open(html_file_path, encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    categories_data: dict[str, Any] = {}

    # Find all main category sections (h2 elements with accordion-head class)
    main_sections = soup.find_all("h2", class_="accordion-head")

    for section in main_sections:
        section_name = section.get_text().strip()
        categories_data[section_name] = {}

        # Find the corresponding accordion body
        accordion_body = section.find_next_sibling("div", class_="accordion-body")
        if not accordion_body or not isinstance(accordion_body, Tag):
            continue

        # Handle Physics section specially due to its nested structure
        if section_name == "Physics":
            categories_data[section_name] = parse_physics_section(accordion_body)
        else:
            # Handle regular sections
            categories_data[section_name] = parse_regular_section(accordion_body)

    return categories_data


def parse_regular_section(accordion_body: Tag) -> dict[str, Any]:
    """Parse regular sections (non-Physics) with standard structure."""
    section_data: dict[str, Any] = {}

    # Find all category entries (h4 elements containing category codes)
    category_entries = accordion_body.find_all("h4")

    for entry in category_entries:
        if not isinstance(entry, Tag):
            continue

        # Extract category code and name
        entry_text = entry.get_text().strip()

        # Pattern to match category code and description
        # e.g., "cs.AI (Artificial Intelligence)" or "q-bio.BM (Biomolecules)"
        match = re.match(r"^([a-z-]+\.[A-Z]+)\s*\(([^)]+)\)", entry_text)

        if match:
            category_code = match.group(1)
            category_name = match.group(2)

            # Find the description paragraph
            description = ""
            # Look for the parent columns div
            parent_columns = entry.find_parent("div", class_="columns")
            if parent_columns and isinstance(parent_columns, Tag):
                # Find all column divs in this row
                columns = parent_columns.find_all("div", class_="column")
                # The description should be in the second column
                if len(columns) >= 2 and isinstance(columns[1], Tag):
                    desc_p = columns[1].find("p")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                    if desc_p and isinstance(desc_p, Tag):
                        description = clean_description(desc_p.get_text())

            section_data[category_code] = {
                "name": category_name,
                "description": description,
            }

    return section_data


def parse_physics_section(accordion_body: Tag) -> dict[str, Any]:
    """Parse the Physics section with its special nested structure."""
    section_data: dict[str, Any] = {}

    # Physics section has subsections like "Astrophysics", "Condensed Matter", etc.
    # Categories are nested under these subsections with h3 headers

    # Find all physics subsection containers (div elements with class "physics columns")
    physics_subsections = accordion_body.find_all("div", class_="physics columns")

    for subsection in physics_subsections:
        if not isinstance(subsection, Tag):
            continue

        # Find all h4 elements containing category codes within this subsection
        category_entries = subsection.find_all("h4")

        for entry in category_entries:
            if not isinstance(entry, Tag):
                continue

            entry_text = entry.get_text().strip()

            # Match various physics category patterns
            patterns = [
                r"^([a-z-]+\.[a-z-]+)\s*\(([^)]+)\)",  # Pattern for categories with hyphens like "cond-mat.dis-nn" or "physics.acc-ph"
                r"^([a-z-]+\.[A-Z-]+)\s*\(([^)]+)\)",  # Standard pattern like "astro-ph.CO"
                r"^([a-z-]+)\s*\(([^)]+)\)",  # Single word categories like "quant-ph"
            ]

            for pattern in patterns:
                match = re.match(pattern, entry_text)
                if match:
                    category_code = match.group(1)
                    category_name = match.group(2)

                    # Find the description
                    description = ""
                    parent_columns = entry.find_parent("div", class_="columns")
                    if parent_columns and isinstance(parent_columns, Tag):
                        columns = parent_columns.find_all("div", class_="column")
                        if len(columns) >= 2 and isinstance(columns[1], Tag):
                            desc_p = columns[1].find("p")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                            if desc_p and isinstance(desc_p, Tag):
                                description = clean_description(desc_p.get_text())

                    section_data[category_code] = {
                        "name": category_name,
                        "description": description,
                    }
                    break

    return section_data


def convert_arxiv_html_to_json(html_file_path: str, output_json_path: str) -> None:
    """Convert arXiv categories HTML file to JSON format.

    Args:
        html_file_path: Path to input HTML file
        output_json_path: Path for output JSON file
    """
    try:
        categories_data = parse_arxiv_categories_html(html_file_path)

        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(categories_data, json_file, indent=2, ensure_ascii=False)

        print(f"Successfully converted {html_file_path} to {output_json_path}")

        # Print summary statistics
        total_categories = 0
        for section_name, section_data in categories_data.items():
            section_count = len(section_data)
            total_categories += section_count
            print(f"  {section_name}: {section_count} categories")

        print(
            f"Total: {total_categories} categories across {len(categories_data)} main sections"
        )

    except Exception as e:
        print(f"Error converting file: {e}")
        raise


if __name__ == "__main__":
    # Convert the arXiv categories HTML to JSON
    convert_arxiv_html_to_json(
        "data/arxiv/arxiv_categories.html", "arxiv_categories.json"
    )
