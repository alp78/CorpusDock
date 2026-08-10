from corpusdock.contracts import CitationLocator, EvidenceResult, SourceCoverage


def test_evidence_result_serializes_its_locator() -> None:
    locator = CitationLocator(
        source_id="source-001",
        locator_type="pdf_page",
        label="p. 42",
        page=42,
    )
    result = EvidenceResult(
        evidence_id="evidence-001",
        excerpt="A short source-grounded excerpt.",
        citation="Example Author (2026), p. 42",
        locator=locator,
        source_path="data/originals/example.pdf",
        verification_status="source-anchor-confirmed",
        chunk_id="chunk-001",
        anchor_ids=("anchor-001",),
        locators=(locator,),
        source_coverage=SourceCoverage(
            extraction_status="partial",
            unresolved_pdf_pages=(1,),
            warnings=("page 1 has no embedded text",),
        ),
    )

    assert result.to_dict()["locator"]["page"] == 42
    assert result.to_dict()["locators"][0]["page"] == 42
    assert result.to_dict()["source_coverage"]["unresolved_pdf_pages"] == [1]
