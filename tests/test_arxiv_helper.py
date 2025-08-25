import pytest

from taxonomy_generator.corpus.arxiv_helper import get_arxiv_id_from_url


@pytest.mark.parametrize(
    "url,include_version,expected",
    [
        # New format with version
        ("https://arxiv.org/abs/1234.5678v1", True, "1234.5678v1"),
        ("https://arxiv.org/abs/1234.5678v1", False, "1234.5678"),
        ("https://arxiv.org/pdf/1234.5678v2.pdf", True, "1234.5678v2"),
        ("https://arxiv.org/pdf/1234.5678v2.pdf", False, "1234.5678"),
        # New format without version
        ("https://arxiv.org/abs/1234.5678", True, "1234.5678"),
        ("https://arxiv.org/abs/1234.5678", False, "1234.5678"),
        ("https://arxiv.org/pdf/1234.5678.pdf", True, "1234.5678"),
        ("https://arxiv.org/pdf/1234.5678.pdf", False, "1234.5678"),
        # Old format with version
        ("https://arxiv.org/abs/hep-th/0603155v2", True, "hep-th/0603155v2"),
        ("https://arxiv.org/abs/hep-th/0603155v2", False, "hep-th/0603155"),
        ("https://arxiv.org/pdf/hep-th/0603155v1.pdf", True, "hep-th/0603155v1"),
        ("https://arxiv.org/pdf/hep-th/0603155v1.pdf", False, "hep-th/0603155"),
        # Old format without version
        ("https://arxiv.org/abs/hep-th/0603155", True, "hep-th/0603155"),
        ("https://arxiv.org/abs/hep-th/0603155", False, "hep-th/0603155"),
        ("https://arxiv.org/pdf/hep-th/0603155.pdf", True, "hep-th/0603155"),
        ("https://arxiv.org/pdf/hep-th/0603155.pdf", False, "hep-th/0603155"),
        ("https://arxiv.org/pdf/eess.SY/0603155.pdf", False, "eess.SY/0603155"),
        # Different URL paths
        ("https://arxiv.org/html/2301.12345v3", True, "2301.12345v3"),
        ("https://arxiv.org/html/2301.12345v3", False, "2301.12345"),
        ("https://arxiv.org/format/2301.12345v1", True, "2301.12345v1"),
        ("https://arxiv.org/format/2301.12345v1", False, "2301.12345"),
        ("https://arxiv.org/e-print/2301.12345", True, "2301.12345"),
        ("https://arxiv.org/e-print/2301.12345", False, "2301.12345"),
        # Old format with different categories
        ("https://arxiv.org/abs/cond-mat/0603155v1", True, "cond-mat/0603155v1"),
        ("https://arxiv.org/abs/cond-mat/0603155v1", False, "cond-mat/0603155"),
        ("https://arxiv.org/abs/cs.AI/0603155", True, "cs.AI/0603155"),
        ("https://arxiv.org/abs/cs.AI/0603155", False, "cs.AI/0603155"),
        ("https://arxiv.org/abs/cv.AI/0603155v1", False, "cv.AI/0603155"),
        # Edge cases
        ("https://arxiv.org/abs/2301.12345", True, "2301.12345"),
        ("https://arxiv.org/abs/2301.12345", False, "2301.12345"),
        ("https://arxiv.org/pdf/2301.12345.pdf", True, "2301.12345"),
        ("https://arxiv.org/pdf/2301.12345.pdf", False, "2301.12345"),
        # 5-digit paper numbers
        ("https://arxiv.org/abs/2301.12345v1", True, "2301.12345v1"),
        ("https://arxiv.org/abs/2301.12345v1", False, "2301.12345"),
        # Fallback case - just extract any arxiv ID pattern
        ("some text with 2301.12345 in it", True, "2301.12345"),
        ("some text with 2301.12345 in it", False, "2301.12345"),
        # Cases that should return empty string
        ("https://example.com/paper.pdf", True, ""),
        ("https://example.com/paper.pdf", False, ""),
        ("not a url at all", True, ""),
        ("not a url at all", False, ""),
        ("https://arxiv.org/abs/", True, ""),
        ("https://arxiv.org/abs/", False, ""),
        ("https://github.com/some/repo", True, ""),
        ("https://github.com/some/repo", False, ""),
        ("random text without arxiv id", True, ""),
        ("random text without arxiv id", False, ""),
        ("", True, ""),
        ("", False, ""),
        ("https://arxiv.org/pdf/invalid-format", True, ""),
        ("https://arxiv.org/pdf/invalid-format", False, ""),
    ],
)
def test_get_arxiv_id_from_url(url: str, include_version: bool, expected: str):
    assert get_arxiv_id_from_url(url, include_version) == expected
