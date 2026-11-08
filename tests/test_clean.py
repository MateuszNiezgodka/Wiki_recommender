import pytest

from wiki_recommender.preprocessing.clean import (
    clean_section_label,
    clean_wiki_text,
    first_paragraph,
    informative_categories,
    split_lead_and_body,
)


@pytest.mark.parametrize(
    "line",
    [
        "IPA help",
        "Musical scores are temporarily disabled due to a security issue. See Help:Score.",
        "%5B%5BWikipedia%3ATemplates+for+discussion%2FLog%5D%5D{{{CRITERION}}}",
        "{{{1}}} is unfilled",
    ],
)
def test_boilerplate_lines_are_removed(line):
    assert clean_wiki_text(f"Kept prose.\n{line}\nMore prose.") == "Kept prose.\nMore prose."


def test_inline_markup_is_removed_and_paragraphs_stay_apart():
    text = 'See https://x.org/a and www.y.org now.\n<span id="a">Tagged</span> [[word]] &amp; more\n\n  '
    assert clean_wiki_text(text) == "See and now.\nTagged word & more"


def test_prose_about_percentages_and_comparisons_is_kept():
    text = "Growth of 5% was 10% below a<b expectations."
    assert clean_wiki_text(text) == text


def test_definition_is_the_first_paragraph():
    definition = (
        "Dog is a domesticated descendant of wolves that lives with humans in many parts of the world."
    )
    assert first_paragraph(f"{definition}\nSecond.") == definition


def test_definition_skips_a_fragment_left_by_a_removed_formula():
    sentence = (
        "The Collatz conjecture is a famous open problem in mathematics about very simple integer sequences."
    )
    assert first_paragraph(f"For even numbers, divide by 2;\n{sentence}") == sentence


def test_definition_of_a_short_lead_is_the_whole_lead():
    assert first_paragraph("Short one.\nShort two.") == "Short one.\nShort two."


def test_lead_is_split_from_the_body():
    assert split_lead_and_body("Lead.\nBody one.\nBody two.", "Lead.") == ("Lead.", "Body one.\nBody two.")


def test_unexpected_summary_leaves_the_text_as_body():
    assert split_lead_and_body("Body text.", "Other lead.") == ("Other lead.", "Body text.")


def test_administrative_categories_are_dropped():
    categories = [
        "Dogs",
        "Living people",
        "Main topic articles",
        "Wikipedia articles published in peer-reviewed literature",
        "Externally peer reviewed articles",
        "Articles of Confederation",
    ]
    assert informative_categories(categories) == ["Dogs", "Articles of Confederation"]


def test_section_label_loses_its_html_anchor():
    assert clean_section_label('<span id="Sexuality"></span>Sexuality and gender') == "Sexuality and gender"
