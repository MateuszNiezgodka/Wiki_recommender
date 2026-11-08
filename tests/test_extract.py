import pytest

from wiki_recommender.crawler.extract import clean_text, extract_article

PAGE = """
<div class="mw-content-ltr mw-parser-output">
  <div class="shortdescription">Short description text</div>
  <style>.mw-parser-output .ambox{border:1px}</style>
  <div role="note" class="hatnote">For other uses, see Dog (disambiguation).</div>
  <table class="infobox"><tr><td>Infobox secret</td></tr></table>
  <p>The <b>dog</b> <span class="IPA">/dɒɡ/</span> is a <a href="/wiki/Domestication">domesticated</a>
     <a href="/wiki/Canidae" class="mw-redirect">canid</a>.<sup class="reference">
     <a href="#c1">[1]</a></sup></p>
  <figure><figcaption>Caption secret</figcaption></figure>
  <div class="mw-heading mw-heading2">
    <h2 id="Biology">Biology</h2>
    <span class="mw-editsection">[<a href="/w/index.php?action=edit">edit</a>]</span>
  </div>
  <p>Area is <span class="mwe-math-element"><math>{\\displaystyle x}</math></span> large
     <a href="/wiki/File:Dog.jpg">image</a> and <a href="/wiki/Wolf#Range">wolf</a>.</p>
  <div class="mw-heading mw-heading3"><h3>Senses</h3></div>
  <ul><li>Smell <ul><li>nested</li></ul></li><li>Hearing</li></ul>
  <blockquote><p>A quoted sentence.</p></blockquote>
  <div class="mw-heading mw-heading2"><h2>See also</h2></div>
  <ul><li><a href="/wiki/Cat">Cat</a></li></ul>
  <div class="mw-heading mw-heading3"><h3>Nested under see also</h3></div>
  <p>Skipped subsection.</p>
  <div class="mw-heading mw-heading2"><h2>References</h2></div>
  <div class="reflist"><ol class="references"><li>Reference secret</li></ol></div>
  <div class="mw-heading mw-heading2"><h2>Legacy</h2></div>
  <p>A section after the excluded ones.</p>
  <div class="navbox"><a href="/wiki/Navbox_link">Navbox secret</a></div>
</div>
"""


def test_keeps_prose_and_drops_boilerplate():
    text = extract_article(PAGE).text
    for kept in [
        "The dog is a domesticated canid.",
        "Smell nested",
        "Hearing",
        "A quoted sentence.",
        "after the excluded",
    ]:
        assert kept in text
    for dropped in [
        "secret",
        "edit",
        "displaystyle",
        "Short description",
        "For other uses",
        "[1]",
        "/dɒɡ/",
        "Skipped",
    ]:
        assert dropped not in text


def test_section_headings_are_separate_from_the_text():
    article = extract_article(PAGE)
    assert article.sections == ["Biology", "Senses", "Legacy"]
    assert "Biology" not in article.text


def test_summary_is_the_lead_only():
    assert extract_article(PAGE).summary == "The dog is a domesticated canid."


def test_outlinks_come_only_from_kept_text():
    assert extract_article(PAGE).outlinks == ["Domestication", "Canidae", "Wolf"]


def test_inline_markup_does_not_split_words():
    assert extract_article('<div class="mw-parser-output"><p>wo<b>rd</b></p></div>').text == "word"


def _body(html):
    return extract_article(f'<div class="mw-parser-output">{html}</div>')


@pytest.mark.parametrize(
    "line",
    [
        "Smith, J. (1998). A Book. Penguin. ISBN 978-0340128459.",
        "Jones (2004). Paper. Nature. 1 (3): 44. doi:10.1017/S0020743802003021.",
        "Albertini, Luigi (1953). Origins of the War. Oxford. OCLC 168712.",
        "Report. Archived from the original on October 30, 2014.",
        "Exhibit. National Museum. 3 January 2024. Retrieved 14 September 2026.",
        "See also: derived algebraic geometry.",
    ],
)
def test_bibliography_lines_are_dropped_with_their_links(line):
    article = _body(f'<ul><li>Kept item</li><li><a href="/wiki/ISBN_(identifier)">{line}</a></li></ul>')
    assert article.text == "Kept item"
    assert article.outlinks == []


@pytest.mark.parametrize(
    "line",
    [
        "An ISBN identifies a book, and doi names identify other documents.",
        "The archive was retrieved from the original site by a volunteer.",
        "See also the later chapters of this work.",
    ],
)
def test_prose_about_the_same_words_is_kept(line):
    assert _body(f"<p>{line}</p>").text == line


def test_long_paragraph_with_a_stray_citation_is_kept():
    paragraph = "Prose sentence with many words in it. " * 20 + "Retrieved 14 September 2026."
    assert "Prose sentence" in _body(f"<p>{paragraph}</p>").text


def test_mediawiki_error_messages_are_removed():
    html = (
        '<p>If proved,<span class="error mw-ext-cite-error">Cite error: A <code>&lt;ref&gt;</code> tag '
        'is missing (see the <a href="/wiki/Help:Cite_errors">help page</a>).</span> this holds.</p>'
    )
    assert _body(html).text == "If proved, this holds."


@pytest.mark.parametrize("heading", ["General and cited references", "Sources and further reading"])
def test_compound_reference_headings_are_excluded(heading):
    article = _body(f"<h2>Body</h2><p>Prose.</p><h2>{heading}</h2><ul><li>A book without an id</li></ul>")
    assert article.sections == ["Body"]
    assert article.text == "Prose."


@pytest.mark.parametrize("heading", ["Literature", "Sources", "Primary sources", "Publications", "Notes"])
def test_ambiguous_sections_keep_prose_and_drop_lists(heading):
    article = _body(
        f"<h2>Culture</h2><p>Intro.</p><h3>{heading}</h3><p>A national epic shaped the language.</p>"
        "<ul><li>A book without an identifier</li></ul>"
    )
    assert article.text == "Intro.\nA national epic shaped the language."
    assert article.sections == ["Culture", heading]


def test_ambiguous_section_holding_only_a_list_leaves_no_trace():
    article = _body("<h2>Body</h2><p>Prose.</p><h2>Literature</h2><ul><li>Some book</li></ul>")
    assert article.text == "Prose."
    assert article.sections == ["Body"]


def test_section_after_an_ambiguous_one_is_kept_whole():
    article = _body("<h2>Sources</h2><ul><li>A book</li></ul><h2>Legacy</h2><ul><li>Kept item</li></ul>")
    assert article.text == "Kept item"
    assert article.sections == ["Legacy"]


def test_picture_gallery_is_museum_content():
    article = _body("<h2>Picture gallery</h2><p>Notable works include portraits.</p>")
    assert article.sections == ["Picture gallery"]
    assert article.text == "Notable works include portraits."


def test_clean_text_repairs_gaps_left_by_removed_elements():
    assert clean_text("Paris ( ; born 1950) was , in  fact[12] [citation needed] ( ) here .") == (
        "Paris (born 1950) was, in fact here."
    )
